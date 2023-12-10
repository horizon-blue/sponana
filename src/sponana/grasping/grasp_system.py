import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from manipulation import running_as_notebook
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.scenarios import AddFloatingRpyJoint, AddRgbdSensors, ycb
from manipulation.utils import ConfigureParser
from pydrake.all import (
    AbstractValue,
    AddMultibodyPlantSceneGraph,
    Concatenate,
    Context,
    DiagramBuilder,
    JointSliders,
    LeafSystem,
    Meshcat,
    MeshcatPoseSliders,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    PiecewisePose,
    PointCloud,
    RandomGenerator,
    Rgba,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    State,
    UniformlyRandomRotationMatrix,
)

import sponana.grasping.grasping_models as grasping_models
import sponana.utils
from sponana.controller.inverse_kinematics import q_nominal_arm, solve_ik
from sponana.grasping.grasp_generator import (
    BananaSystem,
    GenerateAntipodalGraspCandidate,
    ScoreSystem,
    get_unified_point_cloud,
)

logger = logging.getLogger(__name__)

# Transform from the gripper frame (as in the WSG) to the arm frame
# (the frame of the arm link in the Spot URDF)
X_GA = RigidTransform(RollPitchYaw([1.57, 0, 1.57]), np.array([0.0, -0.08, 0.00]))

# GRASP_POSE_AT_00 = RigidTransform(RollPitchYaw([0, 0, np.pi]).ToRotationMatrix()) @ RigidTransform(
#     R=RotationMatrix([
#         [-0.9998433965693921, -0.007765899189127902, 0.015901985610781485],
#         [-0.017523433802785558, 0.3089304559815118, -0.9509231844028279],
#         [0.0024721659209266797, -0.9510529239620893, -0.3090181616326456],
#     ]),
#     p=[0.021171126217858147, 0.09060437151478618, 0.17440060965502766],
# )
GRASP_POSE_AT_00 = RigidTransform(
    R=RotationMatrix([
        [-1, 0, 0],
        [0, 0, -1],
        [0, -1, 0],
    ]),
    p=[0.021171126217858147, 0, 0.17440060965502766],
) @ RigidTransform(RollPitchYaw([0, 3 * np.pi/2, 0]).ToRotationMatrix())

@dataclass
class GraspPlan:
    traj_X_G: any = None
    gripper_frames: list = None
    times: list = None

class Grasper(LeafSystem):
    """
    This system implements banana grasping.

    Constructor args:
    - target_obj_path: The path to the SDF or URDF file for the target object.
    - target_obj_link: The base link name of the target object.
    - target_obj_rpy_str: A string describing an orientation of the target object
        in which a valid grasp can be found.  (Should be in contact with the floor/table
        on the same face as the target object in the environment.)  Of the form [R P Y]
        where R P and Y are numbers in degrees for roll pitch yaw.
    - time_step: The time step at which to update the system.
    - arm_name: The name of the arm link in the Spot URDF.
    - meshcat: The meshcat instance to use for visualization.
    - verbose: Whether to print debugging info.

    Input ports:
    - spot_state: The current state of the Spot robot.
    - banana_pose: The current pose of the banana.
    - do_grasp: Should change from 0 to 1 at the moment we should begin
        to attempt grasping.

    Output ports:
    - arm_position: The next arm position to go to. (Subsequent
        arm positions will be nearby.)
    - banana_grasped: Is the banana grasped

    State:
    - init_time: The time at which grasping began.  (This, plus the time from
        the context, allows the system to tell where in the grasping trajectory
        it is.)
    - grasp_plan: A GraspPlan object describing the trajectory for the spot
        gripper.
    - arm_position: The 7DOF arm position the grasp system has computed to move the
        arm to, to accomplish the current point in the grasp plan.
    - _banana_grasped: Bool, whether the banana has been grasped yet.
    """

    def __init__(
        self,
        # target_obj_path="package://sponana/banana/banana.sdf",
        # target_obj_link="banana",
        # target_obj_rpy_str="[0, 0, 0]",
        target_obj_path="package://sponana/grasping/004_sugar_box.sdf", #"package://sponana/banana/banana.sdf",
        target_obj_link="base_link_sugar", # "banana",
        target_obj_rpy_str="[90., 90., 90.]", # "[0, 0, 0]",
        time_step: float = 0.003,
        arm_name="arm_link_wr1",
        meshcat: Optional[Meshcat] = None,
        verbose=False,
    ):
        super().__init__()

        self.target_obj_path = target_obj_path
        self.target_obj_link = target_obj_link
        self.target_obj_rpy_str = target_obj_rpy_str
        self.arm_name = arm_name
        self.meshcat = meshcat
        self.verbose = verbose

        # MultibodyPlant in the robot's head, used for IK and FK
        self.plant = None
        self.plant_context = None

        self.DeclareInitializationUnrestrictedUpdateEvent(self.first_initialization)
        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec=time_step, offset_sec=0.0, update=self._update
        )

        self.DeclareVectorInputPort("spot_state", 20)
        self.DeclareVectorInputPort("do_grasp", 1)
        self.DeclareAbstractInputPort(
            "banana_pose", AbstractValue.Make(RigidTransform())
        )

        self._banana_grasped = self.DeclareDiscreteState(1)
        self._arm_position = self.DeclareDiscreteState(7)
        self._grasp_plan = self.DeclareAbstractState(AbstractValue.Make(GraspPlan()))
        self._init_time = self.DeclareDiscreteState(1)

        self.DeclareStateOutputPort("arm_position", self._arm_position)
        self.DeclareStateOutputPort("banana_grasped", self._banana_grasped)

    ### State setters and getters ###
    def _set_banana_grasped(self, state, new_val):
        state.get_mutable_discrete_state().set_value(self._banana_grasped, [new_val])

    def _set_arm_position(self, state, new_val):
        state.get_mutable_discrete_state().set_value(self._arm_position, new_val)

    def _set_grasp_plan(self, state, new_val):
        state.get_mutable_abstract_state(self._grasp_plan).set_value(new_val)

    def _set_init_time(self, state, new_val):
        state.get_mutable_discrete_state().set_value(self._init_time, [new_val])

    def _get_init_time(self, state):
        return state.get_mutable_discrete_state(self._init_time).value()
    
    def _get_grasp_plan(self, state):
        return state.get_abstract_state(self._grasp_plan).get_value()

    ### Input port getters ###
    
    def get_spot_state_input_port(self):
        return self.GetInputPort("spot_state")

    def get_banana_pose_input_port(self):
        return self.GetInputPort("banana_pose")

    def get_do_grasp_input_port(self):
        return self.GetInputPort("do_grasp")
    
    ### Output port getters ###

    def get_banana_grasped_output_port(self):
        return self.GetOutputPort("banana_grasped")

    def get_arm_position_output_port(self):
        return self.GetOutputPort("arm_position")

    ### Input getters with preprocessing ###

    def get_current_arm_pose(self, context: Context):
        current_q = self.get_spot_state_input_port().Eval(context)[:10]
        self.plant.SetPositions(self.plant_context, current_q)
        arm_body_idx = self.plant.GetBodyByName(self.arm_name).index()
        X_WA = self.plant.get_body_poses_output_port().Eval(self.plant_context)[
            arm_body_idx
        ]
        return X_WA

    def get_current_gripper_pose(self, context: Context):
        X_WA = self.get_current_arm_pose(context)
        X_WG = X_WA @ X_GA.inverse()
        return X_WG

    def _should_grasp(self, context: Context):
        if not self.get_do_grasp_input_port().HasValue(context):
            # if this port is not connected, then we should always grasp
            # this allows us to test grasping without the FSM
            return True
        return bool(self.get_do_grasp_input_port().Eval(context)[0])

    ### Initialization ###

    def first_initialization(self, context: Context, state: State):
        # Initialize state
        current_arm_position = self.get_spot_state_input_port().Eval(context)[3:10]
        self._set_banana_grasped(state, 0)
        self._set_init_time(state, -np.inf)
        self._set_arm_position(state, current_arm_position)
        self._set_grasp_plan(state, GraspPlan())

        # Begin grasping if `do_grasp` is true
        self._maybe_begin_grasping(context, state)
    
    def _maybe_begin_grasping(self, context: Context, state: State):
        if not self._should_grasp(context):
            return

        if self.verbose:
            print("Beginning grasp sequence.")

        time = context.get_time()
        self._set_init_time(state, time)

        banana_pose = self.get_banana_pose_input_port().Eval(context)
        logger.info(f"Banana pose: {banana_pose}")

        # X_Gs = sample_grasps(
        #     self.target_obj_path,
        #     self.target_obj_link,
        #     self.target_obj_rpy_str,
        #     pointcloud_transform=banana_pose,
        #     meshcat_for_final_grasp=self.meshcat,
        #     meshcat=self.meshcat,
        # )
        best_gripper_pose = banana_pose @ GRASP_POSE_AT_00
        AddMeshcatTriad(
            self.meshcat, f"grasping/Goal gripper pose",
            length=0.25, X_PT=best_gripper_pose, opacity=1
        )

        # Set up mental model for IK
        self.plant, mental_sim_context = get_mental_plant(
            self.target_obj_path,
            self.target_obj_link,
            meshcat=None,  # If we give the mental model the meshcat it will take over the viz
        )
        self.plant_context = self.plant.GetMyContextFromRoot(mental_sim_context)
        # best_gripper_pose = X_Gs[0]  # @ RigidTransform([0., 0., 0.06])

        # Plan a gripper trajectory
        X_WGinitial = self.get_current_gripper_pose(context)
        gripper_frames, times = MakeGripperFrames(
            X_WGinitial, best_gripper_pose, 0.0
        )
        if self.verbose:
            print(gripper_frames)
            print(times)

        traj_X_G = MakeGripperPoseTrajectories(gripper_frames, times)

        # Set the grasp plan in the state
        grasp_plan = GraspPlan(traj_X_G, gripper_frames, times)
        self._set_grasp_plan(state, grasp_plan)

        if self.verbose and self.meshcat is not None:
            for t in np.linspace(0, 7, 70):
                p = traj_X_G.value(t)
                AddMeshcatTriad(
                    self.meshcat, f"grasping/arm trajectory {t}",
                    length=0.125, X_PT=p, opacity=0.25
                )

        # Set the output port for the first time
        self._update(context, state)

    ### Update ###

    def _update(self, context: Context, state: State):
        if not self._should_grasp(context):
            return
        
        if self._get_init_time(state) < 0:
            # If we should grasp, and we haven't initialized -- initialize grasping!
            self._maybe_begin_grasping(context, state)
        
        grasp_plan = self._get_grasp_plan(state)
        times = grasp_plan.times
        traj_X_G = grasp_plan.traj_X_G

        # Gripper pose to go to now
        T = context.get_time() - self._get_init_time(state)
        X_WGnow = RigidTransform(traj_X_G.value(T))

        # Gripper is currently closing
        if times["pick"] < T < times["postpick"]:
            time_fraction = (T - times["pick"]) / (
                times["postpick"] - times["pick"]
            )
            X_WGnow = RigidTransform(traj_X_G.value(times["pick"]))
            _gripper_angle = -1.4 * (1 - time_fraction)
        if T <= times["pick"]:
            _gripper_angle = -1.4
        if T >= times["postpick"]:
            _gripper_angle = -0.24

        # Run IK
        arm_position, ik_success = _run_ik(
            X_WGnow,
            self.plant,
            self.plant_context,
            self.get_spot_state_input_port().Eval(context)[:10],
            self.arm_name,
        )
        if not ik_success and self.verbose:
            print(f"IK failed at T={T}")

        # Move the arm to the next position
        arm_position[-1] = _gripper_angle
        self._set_arm_position(state, arm_position)

        if T > times["postpick2"]:
            self._set_banana_grasped(state, 1)

        return

#####
def _run_ik(X_WG, plant, plant_context, initial_q, arm_frame_name):
    X_WA = X_WG @ X_GA
    soln = solve_ik(
        plant,
        plant_context,
        X_WA,
        fix_base=True,
        base_position=initial_q[:3],
        position_bound=0.01,
        rotation_bound=0.01,
        target_frame_name=arm_frame_name,
        error_on_fail=False,
        q_current=initial_q[3:10],
    )
    if soln is None:
        return (initial_q[3:10], False)
    arm_position = soln[3:]
    return (arm_position, True)


#####


def MakeGripperFrames(X_WGinit, X_WGfinal, t0):
    """
    Here, G is the gripper frame AS THOUGH IT'S THE WSG! (NOT THE LINK FRAME)
    """
    X_WG = {"initial": X_WGinit}
    X_WG["post_init"] = X_WGinit @ RigidTransform([0.0, 0.3, 0.3])
    # X_WG["prepick"] =  X_WGfinal @ RigidTransform(RollPitchYaw([0., -0.6, 0.]), [0., -0.3, 0.05])
    # X_WG["pick"] = X_WGfinal @ RigidTransform(RollPitchYaw([0., -0.6, 0.]), [0., -0.2, 0.05])
    # X_WG["postpick"] = X_WGfinal @ RigidTransform(RollPitchYaw([0., -0.6, 0.]), [0., -0.22, 0.05])
    # X_WG["postpick2"] = X_WGfinal @ RigidTransform([0., -0.3, 0.0])# RigidTransform([0., -0.3, 0.])
    X_WG["prepick"] = X_WGfinal @ RigidTransform([0.0, -0.3, 0.0])
    X_WG["pick"] = X_WGfinal @ RigidTransform(
        [0.0, 0, 0.0]
    )  # @ RigidTransform([0., -0.22, 0.00])
    X_WG["postpick"] = X_WGfinal @ RigidTransform(
        [0.0, 0, 0.0]
    )  # @ RigidTransform([0., -0.22, 0.00])
    X_WG["postpick2"] = X_WGfinal @ RigidTransform(
        [0.0, -0.45, 0.0]
    )  # RigidTransform([0., -0.3, 0.])

    times = {"initial": t0}
    times["post_init"] = times["initial"] + 1.0  # 1 sec to here
    times["prepick"] = times["post_init"] + 2.0  # 2 secs to prepick
    times["pick"] = times["prepick"] + 1.0  # 1 sec to pick spot
    times["postpick"] = times["pick"] + 2.0  # 2 secs to close gripper
    times["postpick2"] = times["postpick"] + 2.0  # 2 secs to next spot

    return X_WG, times


def MakeGripperPoseTrajectories(X_WG, times):
    sample_times = []
    poses = []
    for name in ["initial", "post_init", "prepick", "pick", "postpick", "postpick2"]:
        sample_times.append(times[name])
        poses.append(X_WG[name])
    return PiecewisePose.MakeLinear(sample_times, poses)


######


def make_internal_gripper_model(
    target_obj_path, target_obj_link, target_obj_rpy_str, meshcat=None
):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    sponana.utils.configure_parser(parser)
    parser.AddModelsFromString(
        grasping_models.gripper_and_target_str(
            target_obj_path=target_obj_path,
            target_obj_link=target_obj_link,
            target_obj_rpy_str=target_obj_rpy_str,
        ),
        "dmd.yaml",
    )
    # parser.AddModelsFromUrl("package://sponana/grasping/banana_and_spot_gripper.dmd.yaml")
    plant.Finalize()

    if meshcat is not None:
        params = MeshcatVisualizerParams()
        params.prefix = "planning"
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat, params
        )

    return builder.Build()


from manipulation.scenarios import AddMultibodyTriad


# For visualization
def draw_grasp_candidate(
    X_G, meshcat, gripper_name, prefix="gripper", draw_frames=True
):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    sponana.utils.configure_parser(parser)
    parser.AddModelsFromUrl("package://sponana/grasping/spot_gripper.urdf")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName(gripper_name), X_G)
    plant.Finalize()

    # frames_to_draw = {"gripper": {"body"}} if draw_frames else {}
    params = MeshcatVisualizerParams()
    params.prefix = prefix
    params.delete_prefix_on_initialization_event = False
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)
    AddMultibodyTriad(plant.GetFrameByName(gripper_name), scene_graph)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)


def get_mental_plant(target_obj_path, target_obj_link, meshcat=None):
    if meshcat is not None:
        meshcat.Delete()

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    sponana.utils.configure_parser(parser)

    parser.AddModelsFromString(
        grasping_models.spot_and_target_str(
            target_obj_path=target_obj_path, target_obj_link=target_obj_link
        ),
        "dmd.yaml",
    )
    # parser.AddModelsFromUrl("package://sponana/grasping/banana_and_spot.dmd.yaml")

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
    plant.GetMyContextFromRoot(context)
    scene_graph.GetMyContextFromRoot(context)

    return plant, context


def sample_grasps(
    target_obj_path,
    target_obj_link,
    target_obj_rpy_str,
    gripper_name="gripper",
    pointcloud_transform=RigidTransform(),
    meshcat=None,
    meshcat_for_final_grasp=None,
):
    # meshcat = None
    rng = np.random.default_rng()

    environment = BananaSystem(
        grasping_models.target_and_cameras_str(
            target_obj_path=target_obj_path,
            target_obj_link=target_obj_link,
            target_obj_rpy_str=target_obj_rpy_str,
        )
    )
    environment_context = environment.CreateDefaultContext()
    cloud = get_unified_point_cloud(environment, environment_context, meshcat=meshcat)
    cloud.mutable_xyzs()[:] = pointcloud_transform.multiply(cloud.xyzs())

    if meshcat is not None:
        meshcat.SetObject("planning/cloud", cloud, point_size=0.003)

    internal_model = make_internal_gripper_model(
        target_obj_path, target_obj_link, target_obj_rpy_str, meshcat=meshcat
    )
    internal_model_context = internal_model.CreateDefaultContext()
    costs = []
    X_Gs = []
    for i in range(1000 if running_as_notebook else 2):
        cost, X_G = GenerateAntipodalGraspCandidate(
            internal_model,
            internal_model_context,
            cloud,
            rng,
            wsg_body_index=internal_model.GetSubsystemByName("plant")
            .GetBodyByName(gripper_name)
            .index(),
        )
        if np.isfinite(cost):
            costs.append(cost)
            X_Gs.append(X_G)

    indices = np.asarray(costs).argsort()[:1]
    if meshcat_for_final_grasp is not None:
        for rank, index in enumerate(indices):
            draw_grasp_candidate(
                X_Gs[index], meshcat_for_final_grasp, gripper_name, prefix=f"{rank}th best", draw_frames=False
            )

    return np.array(X_Gs)[indices]
