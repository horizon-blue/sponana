from manipulation.meshcat_utils import AddMeshcatTriad
from pydrake.all import Meshcat, RigidTransform, RotationMatrix

from .rrt import ConfigType


def visualize_path(path: list[ConfigType], meshcat: Meshcat, prefix: int = 0):
    for i, pose in enumerate(path):
        pose = RigidTransform(RotationMatrix.MakeZRotation(pose[2]), [*pose[:2], 0.0])
        opacity = 1 if i == len(path) - 1 else 0.2
        length = 0.3 if i == len(path) - 1 else 0.125
        AddMeshcatTriad(
            meshcat,
            f"trajectory/{prefix}/{i}",
            X_PT=pose,
            opacity=opacity,
            length=length,
        )


def delete_path_visual(path: list[ConfigType], meshcat: Meshcat, prefix: int = 0):
    for i in range(len(path)):
        meshcat.Delete(f"trajectory/{prefix}/{i}")
