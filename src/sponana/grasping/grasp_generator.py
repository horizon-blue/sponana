import numpy as np
from IPython.display import clear_output
from pydrake.all import (
    AbstractValue,
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
)

from manipulation import running_as_notebook
from manipulation.scenarios import AddFloatingRpyJoint, AddRgbdSensors, ycb
from manipulation.utils import ConfigureParser

import sponana.utils

def BananaSystem():
    builder = DiagramBuilder()

    # Create the physics engine + scene graph.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    sponana.utils.configure_parser(parser)
    ConfigureParser(parser)
    parser.AddModelsFromUrl(
        "package://sponana/grasping/banana_and_cameras.dmd.yaml"
    )
    plant.Finalize()

    # Add a visualizer just to help us see the object.
    use_meshcat = False
    if use_meshcat:
        meshcat = builder.AddSystem(MeshcatVisualizer(scene_graph))
        builder.Connect(
            scene_graph.get_query_output_port(),
            meshcat.get_geometry_query_input_port(),
        )

    AddRgbdSensors(builder, plant, scene_graph)

    diagram = builder.Build()
    diagram.set_name("depth_camera_demo_system")
    return diagram

### Point cloud processing ###
def get_unified_point_cloud(system, context, meshcat):
    plant = system.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)

    if meshcat is not None:
        meshcat.Delete()
        meshcat.SetProperty("/Background", "visible", False)

    pcd = []
    for i in range(3):
        cloud = system.GetOutputPort(f"camera{i}_point_cloud").Eval(context)

        if meshcat is not None:
            meshcat.SetObject(f"pointcloud{i}", cloud, point_size=0.001)
            meshcat.SetProperty(f"pointcloud{i}", "visible", False)

        # Crop to region of interest.
        pcd.append(
            cloud.Crop(lower_xyz=[-0.3, -0.3, -0.3], upper_xyz=[0.3, 0.3, 0.3])
        )
        if meshcat is not None:
            meshcat.SetObject(f"pointcloud{i}_cropped", pcd[i], point_size=0.001)
            meshcat.SetProperty(f"pointcloud{i}_cropped", "visible", False)

        pcd[i].EstimateNormals(radius=0.1, num_closest=30)

        camera = plant.GetModelInstanceByName(f"camera{i}")
        body = plant.GetBodyByName("base", camera)
        X_C = plant.EvalBodyPoseInWorld(plant_context, body)
        pcd[i].FlipNormalsTowardPoint(X_C.translation())

    # Merge point clouds.  (Note: You might need something more clever here for
    # noisier point clouds; but this can often work!)
    merged_pcd = Concatenate(pcd)
    if meshcat is not None:
        meshcat.SetObject("merged", merged_pcd, point_size=0.001)

    # Voxelize down-sample.  (Note that the normals still look reasonable)
    down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)
    if meshcat is not None:
        meshcat.SetObject("down_sampled", down_sampled_pcd, point_size=0.001)
        meshcat.SetLineSegments(
            "down_sampled_normals",
            down_sampled_pcd.xyzs(),
            down_sampled_pcd.xyzs() + 0.01 * down_sampled_pcd.normals(),
        )

    return down_sampled_pcd

####### Score System #########
# TODO
def GraspCandidateCost(
    diagram,
    context,
    cloud,
    gripper_body_index=None,
    plant_system_name="plant",
    scene_graph_system_name="scene_graph",
    adjust_X_G=False,
    verbose=False,
    meshcat_path=None,
    meshcat=None
):
    """
    Args:
        diagram: A diagram containing a MultibodyPlant+SceneGraph that contains
            a free body gripper and any obstacles in the environment that we
            want to check collisions against. It should not include the objects
            in the point cloud; those are handled separately.
        context: The diagram context.  All positions in the context will be
            held fixed *except* the gripper free body pose.
        cloud: a PointCloud in world coordinates which represents candidate
            grasps.
        gripper_body_index: The body index of the gripper in plant.  If None, then
            a body named "body" will be searched for in the plant.

    Returns:
        cost: The grasp cost

    If adjust_X_G is True, then it also updates the gripper pose in the plant
    context.
    """
    plant = diagram.GetSubsystemByName(plant_system_name)
    plant_context = plant.GetMyMutableContextFromRoot(context)
    scene_graph = diagram.GetSubsystemByName(scene_graph_system_name)
    scene_graph_context = scene_graph.GetMyMutableContextFromRoot(context)
    if gripper_body_index:
        wsg = plant.get_body(gripper_body_index)
    else:
        assert False, "gripper_body_index is required"

    X_G = plant.GetFreeBodyPose(plant_context, wsg)

    # Transform cloud into gripper frame
    X_GW = X_G.inverse()
    p_GC = X_GW @ cloud.xyzs()

    # Crop to a region inside of the finger box.
    # TODO check this
    crop_min = [-0.05, 0.03, -0.00625]
    crop_max = [0.05, 0.15, 0.00625]
    indices = np.all(
        (
            crop_min[0] <= p_GC[0, :],
            p_GC[0, :] <= crop_max[0],
            crop_min[1] <= p_GC[1, :],
            p_GC[1, :] <= crop_max[1],
            crop_min[2] <= p_GC[2, :],
            p_GC[2, :] <= crop_max[2],
        ),
        axis=0,
    )

    if meshcat_path:
        pc = PointCloud(np.sum(indices))
        pc.mutable_xyzs()[:] = cloud.xyzs()[:, indices]
        meshcat.SetObject(
            "planning/points", pc, rgba=Rgba(1.0, 0, 0), point_size=0.01
        )

    if adjust_X_G and np.sum(indices) > 0:
        p_GC_x = p_GC[0, indices]
        p_Gcenter_x = (p_GC_x.min() + p_GC_x.max()) / 2.0
        X_G.set_translation(X_G @ np.array([p_Gcenter_x, 0, 0]))
        plant.SetFreeBodyPose(plant_context, wsg, X_G)
        X_GW = X_G.inverse()

    query_object = scene_graph.get_query_output_port().Eval(
        scene_graph_context
    )

    # Check collisions between the gripper and the sink
    if query_object.HasCollisions():
        cost = np.inf
        if verbose:
            print("Gripper is colliding with something!\n")
            print(f"cost: {cost}")
        return cost

    # Check collisions between the gripper and the point cloud
    # must be smaller than the margin used in the point cloud preprocessing.
    margin = 0.0
    for i in range(cloud.size()):
        distances = query_object.ComputeSignedDistanceToPoint(
            cloud.xyz(i), threshold=margin
        )
        if distances:
            cost = np.inf
            if verbose:
                print("Gripper is colliding with the point cloud!\n")
                print(f"cost: {cost}")
            return cost

    n_GC = X_GW.rotation().multiply(cloud.normals()[:, indices])

    # Penalize deviation of the gripper from vertical.
    # weight * -dot([0, 0, -1], R_G * [0, 1, 0]) = weight * R_G[2,1]
    cost = 20.0 * X_G.rotation().matrix()[2, 1]

    # Reward sum |dot product of normals with gripper x|^2
    cost -= np.sum(n_GC[0, :] ** 2)

    if verbose:
        print(f"cost: {cost}")
        print(f"normal terms: {n_GC[0,:]**2}")
    return cost

class ScoreSystem(LeafSystem):
    def __init__(self, diagram, cloud, gripper_name, wsg_pose_index, meshcat=None):
        LeafSystem.__init__(self)
        self.meshcat = meshcat
        self._diagram = diagram
        self._context = diagram.CreateDefaultContext()
        self._plant = diagram.GetSubsystemByName("plant")
        self._plant_context = self._plant.GetMyMutableContextFromRoot(
            self._context
        )
        wsg = self._plant.GetBodyByName(gripper_name)
        self._gripper_body_index = wsg.index()
        self._wsg_pose_index = wsg_pose_index
        self._cloud = cloud
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self.DeclareForcedPublishEvent(self.Publish)

    def Publish(self, context):
        X_WG = self.get_input_port(0).Eval(context)[self._wsg_pose_index]
        self._plant.SetFreeBodyPose(
            self._plant_context,
            self._plant.get_body(self._gripper_body_index),
            X_WG,
        )
        GraspCandidateCost(
            self._diagram,
            self._context,
            self._cloud,
            gripper_body_index=self._gripper_body_index,
            verbose=True,
            meshcat_path="planning/cost",
            meshcat=self.meshcat,
        )
        clear_output(wait=True)

### Grasp generator ###

def GenerateAntipodalGraspCandidate(
    diagram,
    context,
    cloud,
    rng,
    wsg_body_index=None,
    plant_system_name="plant",
    scene_graph_system_name="scene_graph",
):
    """
    Picks a random point in the cloud, and aligns the robot finger with the normal of that pixel.
    The rotation around the normal axis is drawn from a uniform distribution over [min_roll, max_roll].
    Args:
        diagram: A diagram containing a MultibodyPlant+SceneGraph that contains
            a free body gripper and any obstacles in the environment that we
            want to check collisions against. It should not include the objects
            in the point cloud; those are handled separately.
        context: The diagram context.  All positions in the context will be
            held fixed *except* the gripper free body pose.
        cloud: a PointCloud in world coordinates which represents candidate
            grasps.
        rng: a np.random.default_rng()
        wsg_body_index: The body index of the gripper in plant.  If None, then
            a body named "body" will be searched for in the plant.

    Returns:
        cost: The grasp cost
        X_G: The grasp candidate
    """
    plant = diagram.GetSubsystemByName(plant_system_name)
    plant_context = plant.GetMyMutableContextFromRoot(context)
    scene_graph = diagram.GetSubsystemByName(scene_graph_system_name)
    scene_graph.GetMyMutableContextFromRoot(context)
    if wsg_body_index:
        wsg = plant.get_body(wsg_body_index)
    else:
        assert False, "wsg_body_index is required"

    index = rng.integers(0, cloud.size() - 1)

    # Use S for sample point/frame.
    p_WS = cloud.xyz(index)
    n_WS = cloud.normal(index)

    assert np.isclose(
        np.linalg.norm(n_WS), 1.0
    ), f"Normal has magnitude: {np.linalg.norm(n_WS)}"

    Gx = n_WS  # gripper x axis aligns with normal
    # make orthonormal y axis, aligned with world down
    y = np.array([0.0, 0.0, -1.0])
    if np.abs(np.dot(y, Gx)) < 1e-6:
        # normal was pointing straight down.  reject this sample.
        return np.inf, None

    Gy = y - np.dot(y, Gx) * Gx
    Gz = np.cross(Gx, Gy)
    R_WG = RotationMatrix(np.vstack((Gx, Gy, Gz)).T)
    p_GS_G = [0.054 - 0.01, 0.10625, 0.1]

    # Try orientations from the center out
    min_roll = -np.pi / 3.0
    max_roll = np.pi / 3.0
    alpha = np.array([0.5, 0.65, 0.35, 0.8, 0.2, 1.0, 0.0])
    for theta in min_roll + (max_roll - min_roll) * alpha:
        # Rotate the object in the hand by a random rotation (around the normal).
        R_WG2 = R_WG.multiply(RotationMatrix.MakeXRotation(theta))

        # Use G for gripper frame.
        p_SG_W = -R_WG2.multiply(p_GS_G)
        p_WG = p_WS + p_SG_W

        X_G = RigidTransform(R_WG2, p_WG)
        plant.SetFreeBodyPose(plant_context, wsg, X_G)
        cost = GraspCandidateCost(diagram, context, cloud, adjust_X_G=True, gripper_body_index=wsg_body_index)
        X_G = plant.GetFreeBodyPose(plant_context, wsg)
        if np.isfinite(cost):
            return cost, X_G

        # draw_grasp_candidate(X_G, f"collision/{theta:.1f}")

    return np.inf, None