from pydrake.all import (
    RigidTransform,
    RotationMatrix,
)
import numpy as np

# all table poses
Xs_WT = [
    RigidTransform(
        R=RotationMatrix([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]),
        p=[0.0, 0.0, 0.19925],
    ),
    RigidTransform(
        R=RotationMatrix([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]),
        p=[0.0, 2.0, 0.19925],
    ),
    RigidTransform(
        R=RotationMatrix([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]),
        p=[0.0, -2.0, 0.19925],
    )
]

camera_poses_W = [
    RigidTransform(
        R=RotationMatrix([
            [-0.008426572055398018, 0.49998224790571183, -0.865994656255191],
            [0.9999644958114239, 0.004213286027699008, -0.007297625466794737],
            [0.0, -0.8660254037844386, -0.49999999999999983],
        ]),
        p=[0.5567144688632728, -0.003735510093671053, 0.495],
    ),
    RigidTransform(
        R=RotationMatrix([
            [0.7564493864543211, -0.32702611735642795, 0.5664258506633156],
            [-0.6540522347128561, -0.3782246932271605, 0.6551043853465944],
            [0.0, -0.8660254037844387, -0.4999999999999999],
        ]),
        p=[-0.4067672805262673, -0.5122634135249003, 0.495],
    ),
    RigidTransform(
        R=RotationMatrix([
            [-0.8214529060279898, -0.28513817842327355, 0.49387381220674975],
            [-0.5702763568465472, 0.4107264530139948, -0.7113990846327904],
            [0.0, -0.8660254037844387, -0.4999999999999999],
        ]),
        p=[-0.35091572089593653, 0.4881919030929625, 0.495],
    )
]
table_pose = Xs_WT[0] # Table these camera poses are around

def get_camera_poses_table_frame():
    X_WT = table_pose
    X_TW = X_WT.inverse()
    # Camera poses in table frame
    Xs_TC = [X_TW @ X_WC for X_WC in camera_poses_W]
    return Xs_TC

def get_all_camera_poses_world_frame():
    Xs_TC = get_camera_poses_table_frame()
    Xs_WC = []
    for X_WT in Xs_WT:
        for X_TC in Xs_TC:
            Xs_WC.append(X_WT @ X_TC)
    return Xs_WC

def get_camera_generator_str():
    Xs_TC = get_camera_poses_table_frame()
    str = ""
    for (table_idx, X_WT) in enumerate(Xs_WT):
        for (camera_idx, X_TC) in enumerate(Xs_TC):
            X_WC = X_WT @ X_TC
            trans = X_WC.translation()
            rot = 180/np.pi * X_WC.rotation().ToRollPitchYaw().vector()
            str += f"""
    camera{camera_idx}_at_table{table_idx}:
        name: "camera{camera_idx}_at_table{table_idx}"
        depth: True
        X_PB:
            translation: [{trans[0]}, {trans[1]}, {trans[2]}]
            rotation: !Rpy {{ deg: [{rot[0]}, {rot[1]}, {rot[2]}] }}
"""
    print(str)
    return str
