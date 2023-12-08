from manipulation.scenarios import ExtractBodyPose
from manipulation.station import ExtractPose, GetScopedFrameByName
from pydrake.all import CameraConfig, Diagram, DiagramBuilder, RotationMatrix, RigidTransform


def add_camera_pose_extractor(
    camera_config: CameraConfig, station: Diagram, builder: DiagramBuilder
):
    plant = station.GetSubsystemByName("plant")
    # frame names in local variables:
    # P for parent frame, B for base frame, C for camera frame.

    # Extract the camera extrinsics from the config struct.
    P = (
        GetScopedFrameByName(plant, camera_config.X_PB.base_frame)
        if camera_config.X_PB.base_frame
        else plant.world_frame()
    )
    X_PC = camera_config.X_PB.GetDeterministicValue()
    X_BP = P.GetFixedPoseInBodyFrame()
    X_BC = X_BP @ X_PC

    # convert mbp frame to geometry frame
    body = P.body()
    camera_pose = builder.AddNamedSystem(
        f"{camera_config.name}.pose",
        ExtractPose(station.GetOutputPort("body_poses"), body.index(), X_BC),
    )
    builder.Connect(
        station.GetOutputPort("body_poses"),
        camera_pose.get_input_port(),
    )
    return camera_pose


def add_body_pose_extractor(
    model_instance_name: str, body_name: str, station: Diagram, builder: DiagramBuilder
):
    plant = station.GetSubsystemByName("plant")
    pose_extractor = builder.AddNamedSystem(
        f"{model_instance_name}.pose",
        ExtractBodyPose(
            station.GetOutputPort("body_poses"),
            plant.GetBodyByName(
                body_name, plant.GetModelInstanceByName(model_instance_name)
            ).index(),
        ),
    )
    builder.Connect(
        station.GetOutputPort("body_poses"), pose_extractor.get_input_port()
    )
    return pose_extractor

def b3d_banana_pose_to_drake(X_W_NewBananaB3d):
    # Offline, I captured the pose of a banana in one simulation,
    # as reported by Bayes3D and by Drake.
    # We can use these as reference points to understand the appropriate
    # transformation to convert between these representations.
    X_W_BananaDrake = RigidTransform( # drake pose
        R=RotationMatrix([
            [-0.0003172594814682128, 0.9998758496538035, -0.015753876523125154],
            [-0.9999552082234492, -0.00046622868823287256, -0.009453262802261178],
            [-0.00945943408559755, 0.015750171741753295, 0.9998312113536392],
        ]),
        p=[0.09965310518406117, 3.999831617167207, 0.22564659809789764],
    )
    X_W_BananaB3d = RigidTransform( # same pose, in bayes3d
        R=RotationMatrix([
            [0.9500731229782104, -0.31202712655067444, 5.150570103751306e-08],
            [0.3120270073413849, 0.9500731229782104, 2.2046666714459207e-08],
            [0.0, -7.450580596923828e-09, 1.0],
        ]),
        p=[0.14940664172172546, 3.996229410171509, 0.22508499026298523],
    )
    
    X_BananaB3d_BananaDrake = X_W_BananaB3d.inverse() @ X_W_BananaDrake
    
    def convert(X_W_NewBananaB3d):
        return X_W_NewBananaB3d @ X_BananaB3d_BananaDrake

    return convert(X_W_NewBananaB3d)