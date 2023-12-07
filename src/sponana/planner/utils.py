from manipulation.meshcat_utils import AddMeshcatTriad
from pydrake.all import Meshcat, RigidTransform, RotationMatrix

from .rrt import ConfigType


def visualize_path(path: list[ConfigType], meshcat: Meshcat):
    for i, pose in enumerate(path):
        pose = RigidTransform(RotationMatrix.MakeZRotation(pose[2]), [*pose[:2], 0.0])
        opacity = 0.2
        AddMeshcatTriad(meshcat, f"trajectory_{i}", X_PT=pose, opacity=opacity)


def delete_path_visual(path: list[ConfigType], meshcat: Meshcat):
    for i in range(len(path)):
        meshcat.Delete(f"trajectory_{i}")
