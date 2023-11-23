from manipulation.scenarios import ExtractBodyPose
from manipulation.station import ExtractPose, GetScopedFrameByName
from pydrake.all import CameraConfig, Diagram, DiagramBuilder


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
