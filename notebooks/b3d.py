import numpy as np
import matplotlib.pyplot as plt
import pickle
from pydrake.all import RigidTransform, RollPitchYaw, RotationMatrix
import jax.numpy as jnp
import sponana.perception.bayes3d.bayes3d_interface as b3d

color_image = np.load("/home/georgematheos/sponana/notebooks/assets/imgs/clutter_1/color_image.npy")
depth_image = np.load("/home/georgematheos/sponana/notebooks/assets/imgs/clutter_1/depth_image.npy")
camera_info = pickle.load(open("/home/georgematheos/sponana/notebooks/assets/camera_info.pkl", "rb"))

categories = ["banana", "cracker_box", "potted_meat_can", "gelatin_box"]

camera_pose = RigidTransform(
  R=RotationMatrix([
    [1.0, 0.0, 0.0],
    [0.0, -0.9659258262890683, 0.2588190451025208],
    [0.0, -0.2588190451025208, -0.9659258262890683],
  ]),
  p=[0.0, 1.7017037086855467, 0.987059047744874],
)

table_top_position = np.array([0, 2.0, -0.02])
table_top_rotation = RotationMatrix()
table_top_pose = RigidTransform(table_top_rotation, table_top_position)

from typing import NamedTuple

class CameraImage(NamedTuple):
    camera_pose: RigidTransform
    intrinsics: np.ndarray # 3 x 3
    color_image: np.ndarray # W x H x 3
    depth_image: np.ndarray # W x H

camera_image = CameraImage(
    camera_pose=camera_pose,
    intrinsics=camera_info.intrinsic_matrix(),
    color_image=color_image[:, :, :3],
    depth_image=depth_image[:, :, 0],
)

def external_pose_to_b3d_pose(rigid_transform):
    return jnp.array(rigid_transform.GetAsMatrix4())

def b3d_pose_to_external_pose(b3d_pose):
    return RigidTransform(b3d_pose)

b3d.b3d_init(
    camera_image,
    categories,
    'banana',
    5, # banana, cracker box, 2 potted meat cans, gelatin box
    (table_top_pose, 0.49, 0.63, 0.015),
    scaling_factor=0.2,
    external_pose_to_b3d_pose=external_pose_to_b3d_pose,
    b3d_pose_to_external_pose=b3d_pose_to_external_pose,
)

print("Done.")