from typing import Optional

import numpy as np
from manipulation.meshcat_utils import AddMeshcatTriad
from pydrake.all import (
    PiecewisePose,
    Context,
    AbstractValue,
    LeafSystem,
    Meshcat,
    RigidTransform,
    RotationMatrix,
    State,
    AddMultibodyPlantSceneGraph,
    Concatenate,
    DiagramBuilder,
    JointSliders,
    LeafSystem,
    MeshcatPoseSliders,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    PointCloud,
    RandomGenerator,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    UniformlyRandomRotationMatrix,
    RollPitchYaw,
)

from manipulation import running_as_notebook
from manipulation.scenarios import AddFloatingRpyJoint, AddRgbdSensors, ycb
from manipulation.utils import ConfigureParser
from sponana.controller.inverse_kinematics import solve_ik
import sponana.utils
from sponana.grasping.grasp_generator import get_unified_point_cloud, BananaSystem, ScoreSystem, GenerateAntipodalGraspCandidate

X_GA = RigidTransform(
        RollPitchYaw([0, 0, 1.57]), np.array([0.0, 0.08, 0.00])
    )

class Grasper(LeafSystem):
    """
    This system implements banana grasping.

    Input ports:
    - spot_state: The current state of the Spot robot.
    - banana_pose: The current pose of the banana.
    - reset_time: The next time at which to reset the system (and read
        in a new banana pose.)

    Output ports:
    - arm_position: The next arm position to go to. (Subsequent
        arm positions will be nearby.)
    """

    def __init__(
        self,
        time_step: float = 0.01,
        arm_name="arm_link_wr1",
        meshcat: Optional[Meshcat] = None
    ):
        super().__init__()

        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=time_step, offset_sec=0.0, update=self._update
        )

        # At first I thought we needed to track the state, but
        # now I think we can just use the time.
        # # Will be
        # # 0 if pre-grasp,
        # # 1 if grasping,
        # # 2 if post-grasp,
        # # 3 if done.
        # self._current_state = self.DeclareDiscreteState(1)

        # Reset things once this time is passed.
        self.DeclareVectorInputPort("reset_time", 1)
        self._last_reset = -np.inf
        
        self.DeclareVectorInputPort("spot_state", 20)
        self.DeclareAbstractInputPort("banana_pose", AbstractValue.Make(RigidTransform()))

        self.DeclareVectorOutputPort("arm_position", 7, self.OutputArmPosition)
        self._arm_position = None   # Value returned by OutputArmPosition
        self._gripper_angle = None  # Last value of _arm_position is overwritten with this
                                    # -1.4 = open gripper

        self.DeclareInitializationDiscreteUpdateEvent(self._initialize)

        self.arm_name = arm_name
        self.meshcat = meshcat

        # MultibodyPlant in the robot's head, used for IK and FK
        self.plant = None
        self.plant_context = None

    def OutputArmPosition(self, context: Context, output):
        arm_pos = self._arm_position.copy()
        arm_pos[-1] = self._gripper_angle
        output.SetFromVector(arm_pos)

    def get_reset_time_input_port(self):
        return self.get_input_port(0)

    def get_spot_state_input_port(self):
        return self.get_input_port(1)

    def get_banana_pose_input_port(self):
        return self.get_input_port(2)
    
    def get_current_arm_pose(self, context: Context):
        current_q = self.get_spot_state_input_port().Eval(context)[:10]
        self.plant.SetPositions(self.plant_context, current_q)
        arm_body_idx = self.plant.GetBodyByName(self.arm_name).index()
        X_WA = self.plant.get_body_poses_output_port().Eval(self.plant_context)[arm_body_idx]
        return X_WA

    def get_current_gripper_pose(self, context: Context):
        X_WA = self.get_current_arm_pose(context)
        X_WG = X_WA @ X_GA.inverse()
        return X_WG

    def _initialize(self, context: Context, state: State):
        # Wait until the reset time to do the first initialization...
        if context.get_time() < self.get_reset_time_input_port().Eval(context)[0] and self._last_reset == -np.inf:
            self._arm_position = self.get_spot_state_input_port().Eval(context)[3:10]
            self._gripper_angle = self._arm_position[-1]
            return

        # Open gripper.
        self._gripper_angle = -1.4  # Last value of _arm_position is overwritten with this

        banana_pose = self.get_banana_pose_input_port().Eval(context)

        print("sampling grasps...")
        X_Gs, self.plant, mental_sim_context = sample_grasps(
            pointcloud_transform=banana_pose,
            meshcat_for_final_grasp=self.meshcat
        )
        # input("Press enter to continue...")
        self.plant_context = self.plant.GetMyContextFromRoot(mental_sim_context)
        best_gripper_pose = X_Gs[0] # @ RigidTransform([0., 0., 0.06])

        X_WGinitial = self.get_current_gripper_pose(context)
        gripper_frames, self.times = MakeGripperFrames(X_WGinitial, best_gripper_pose, self._last_reset)

        self.traj_X_G = MakeGripperPoseTrajectories(gripper_frames, self.times)

        for name, frame in gripper_frames.items():
            AddMeshcatTriad(
                self.meshcat, f"{name}_frame", X_PT=frame
            )
        # for i in range(50):
        #     X_WG = self.traj_X_G.value(i * 0.1)
        #     AddMeshcatTriad(
        #         self.meshcat, f"trajectory_{i * 0.1}", X_PT=X_WG
        #     )

        ### Set the output port for the first time ###
        self._update(context, state)
        
    def _update(self, context: Context, state: State):
        newest_reset_time = self.get_reset_time_input_port().Eval(context)[0]
        reset_already_done = newest_reset_time <= self._last_reset
        if (not reset_already_done) and newest_reset_time < context.get_time():
            print("Re-initializing")
            self._last_reset = newest_reset_time
            self._initialize(context, state)
            return
        elif self._last_reset == -np.inf:
            # Haven't started yet
            return

        # Gripper pose to go to now
        X_WGnow = RigidTransform(self.traj_X_G.value(context.get_time()))

        # Gripper is currently closing
        if self.times["pick"] < context.get_time() and context.get_time() < self.times["postpick"]:
            # assert np.allclose(X_WGnow, self.get_current_gripper_pose(context))
            time_fraction = (context.get_time() - self.times["pick"]) / (self.times["postpick"] - self.times["pick"])
            gripper_angle = -1.4 * (1 - time_fraction)
            self._gripper_angle = gripper_angle
            print(f"T = {context.get_time()} | gripper_angle = {gripper_angle}")
            return

        print(f"T = {context.get_time()} | X_WGnow = {self.traj_X_G.value(context.get_time())}")
        
        # Run IK
        arm_position = _run_ik(X_WGnow, self.plant, self.plant_context, self.get_spot_state_input_port().Eval(context)[:10], self.arm_name)
        print(f"arm_position = {arm_position}")
        # Move the arm to the next position
        self._arm_position = arm_position

#####
def _run_ik(X_WG, plant, plant_context, initial_q, arm_frame_name):
    X_WA = X_WG @ X_GA
    soln = solve_ik(
        plant,
        plant_context,
        X_WA, fix_base=True,
        base_position=initial_q[:3],
        position_bound=0.0001,
        rotation_bound=0.0001,
        target_frame_name=arm_frame_name,
        error_on_fail=False,
        q_current=initial_q[3:10]
    )
    if soln is None:
        print("IK failed.")
        return initial_q[3:10]
    arm_position = soln[3:]
    return arm_position

#####

def MakeGripperFrames(X_WGinit, X_WGfinal, t0):
    """
    Here, G is the gripper frame AS THOUGH IT'S THE WSG! (NOT THE LINK FRAME)
    """
    X_WG = {"initial": X_WGinit}
    X_WG["prepick"] =  X_WGfinal @ RigidTransform(RollPitchYaw([0., -0.6, 0.]), [0., -0.3, 0.05])
    X_WG["pick"] = X_WGfinal @ RigidTransform(RollPitchYaw([0., -0.6, 0.]), [0., -0.2, 0.05])
    X_WG["postpick"] = X_WGfinal @ RigidTransform(RollPitchYaw([0., -0.6, 0.]), [0., -0.22, 0.05])
    X_WG["postpick2"] = X_WGfinal @ RigidTransform([0., -0.3, 0.0])# RigidTransform([0., -0.3, 0.])

    times = {"initial": t0}
    times["prepick"] = times["initial"] + 3.0
    times["pick"] = times["prepick"] + 2.0
    times["postpick"] = times["pick"] + 3.0
    times["postpick2"] = times["postpick"] + 2.0
    
    return X_WG, times

def MakeGripperPoseTrajectories(X_WG, times):
    sample_times = []
    poses = []
    for name in ["initial", "prepick", "pick", "postpick", "postpick2"]:
        sample_times.append(times[name])
        poses.append(X_WG[name])
    return PiecewisePose.MakeLinear(sample_times, poses)

######

def make_internal_model():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    sponana.utils.configure_parser(parser)
    parser.AddModelsFromUrl("package://sponana/grasping/banana_and_spot_gripper.dmd.yaml")
    plant.Finalize()
    return builder.Build()

# For visualization
def draw_grasp_candidate(X_G, meshcat, gripper_name, prefix="gripper", draw_frames=True):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    sponana.utils.configure_parser(parser)
    parser.AddModelsFromUrl(
        "package://sponana/grasping/spot_gripper.urdf"
    )
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName(gripper_name), X_G)
    plant.Finalize()

    # frames_to_draw = {"gripper": {"body"}} if draw_frames else {}
    params = MeshcatVisualizerParams()
    params.prefix = prefix
    params.delete_prefix_on_initialization_event = False
    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, params
    )
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)

def sample_grasps(gripper_name="gripper", pointcloud_transform=RigidTransform(), meshcat=None, meshcat_for_final_grasp=None):
    if meshcat is not None:
        meshcat.Delete()
    rng = np.random.default_rng()

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    sponana.utils.configure_parser(parser)
    
    parser.AddModelsFromUrl("package://sponana/grasping/banana_and_spot.dmd.yaml")

    plant.Finalize()

    if meshcat is not None:
        params = MeshcatVisualizerParams()
        params.prefix = "planning"
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat, params
        )
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)

    # Hide the planning gripper
    if meshcat is not None:
        meshcat.SetProperty("planning/gripper", "visible", False)

    environment = BananaSystem()
    environment_context = environment.CreateDefaultContext()
    cloud = get_unified_point_cloud(
        environment,
        environment_context,
        meshcat=None
    )
    print(f"pointcloud transform: {pointcloud_transform}")
    cloud.mutable_xyzs()[:] = pointcloud_transform.multiply(cloud.xyzs())

    if meshcat is not None:
        meshcat.SetObject("planning/cloud", cloud, point_size=0.003)

    plant.GetMyContextFromRoot(context)
    scene_graph.GetMyContextFromRoot(context)

    internal_model = make_internal_model()
    internal_model_context = internal_model.CreateDefaultContext()
    costs = []
    X_Gs = []
    for i in range(1000 if running_as_notebook else 2):
        cost, X_G = GenerateAntipodalGraspCandidate(
            internal_model, internal_model_context, cloud, rng,
            wsg_body_index=internal_model.GetSubsystemByName("plant").GetBodyByName(gripper_name).index(),
        )
        if np.isfinite(cost):
            costs.append(cost)
            X_Gs.append(X_G)

    indices = np.asarray(costs).argsort()[:1]
    # if meshcat_for_final_grasp is not None:
    #     for rank, index in enumerate(indices):
    #         draw_grasp_candidate(
    #             X_Gs[index], meshcat_for_final_grasp, gripper_name, prefix=f"{rank}th best", draw_frames=False
    #         )

    print(max(costs))

    return np.array(X_Gs)[indices], plant, context
