"""
_bayes3d.py
This file contains lower-level Bayes3D inference code,
which is used by bayes3d.py to implement the Bayes3D interface.
"""

import numpy as np
import jax.numpy as jnp
import jax
import bayes3d as b
from bayes3d.utils.ycb_loader import MODEL_NAMES
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
import trimesh
import os
import glob
import bayes3d.neural
import pickle
# Can be helpful for debugging:
# jax.config.update('jax_enable_checks', True) 
from bayes3d.neural.segmentation import carvekit_get_foreground_mask
import genjax
import pyransac3d

### Convert YCB category names to renderer indices ###
CAT_NAMES = [m[4:] for m in MODEL_NAMES]
def category_name_to_renderer_idx(name):
    if name == 'cube':
        return len(CAT_NAMES)
    elif name == 'pillar':
        return len(CAT_NAMES) + 1
    elif name == 'plane':
        return len(CAT_NAMES) + 2
    elif name == 'table':
        return len(CAT_NAMES) + 3
    return CAT_NAMES.index(name)
def renderer_idx_to_category_name(idx):
    if idx == len(CAT_NAMES):
        return 'cube'
    elif idx == len(CAT_NAMES) + 1:
        return 'pillar'
    elif idx == len(CAT_NAMES) + 2:
        return 'plane'
    elif idx == len(CAT_NAMES) + 3:
        return 'table'
    return CAT_NAMES[idx]

### Utils for setting up the Bayes3D problem ###
def find_plane(point_cloud, threshold, minPoints=100, maxIteration=1000):
    """
    Returns the pose of a plane from a point cloud.
    """
    plane = pyransac3d.Plane()
    plane_eq, inliers = plane.fit(point_cloud, threshold, minPoints=minPoints, maxIteration=maxIteration)
    plane_pose = b.utils.plane_eq_to_plane_pose(plane_eq)
    return plane_pose, inliers


def scale_remove_and_setup_renderer(rgbd, scaling_factor=0.5, table_pose_in_cam_frame=None):    
    rgbd_scaled_down = b.RGBD.scale_rgbd(rgbd, scaling_factor)

    b.setup_renderer(rgbd_scaled_down.intrinsics)

    cloud = b.unproject_depth(rgbd_scaled_down.depth, rgbd_scaled_down.intrinsics).reshape(-1,3)
    too_big_indices = np.where(cloud[:,2] > 1.2)
    cloud = cloud.at[too_big_indices, :].set(np.nan)

    too_small_indices = np.where(cloud[:,2] < 0.1)
    cloud = cloud.at[too_small_indices, :].set(np.nan)

    # if table_pose_in_cam_frame is None:
    table_pose, inliers = find_plane(np.array(cloud), 0.01)
    camera_pose = jnp.eye(4)
    table_pose_in_cam_frame = b.t3d.inverse_pose(camera_pose) @ table_pose
    if table_pose_in_cam_frame[2,2] > 0:
        table_pose = table_pose @ b.t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi)
    # else:
    #     table_pose = table_pose_in_cam_frame

    # Depth image, cropping out table, too far, and too close points
    depth_im = jnp.array(rgbd_scaled_down.depth)
    x_indices, y_indices = np.unravel_index(inliers, depth_im.shape) 
    depth_im = depth_im.at[x_indices, y_indices].set(b.RENDERER.intrinsics.far)
    x_indices, y_indices = np.unravel_index(too_big_indices, depth_im.shape)
    depth_im = depth_im.at[x_indices, y_indices].set(b.RENDERER.intrinsics.far)
    x_indices, y_indices = np.unravel_index(too_small_indices, depth_im.shape)
    depth_im = depth_im.at[x_indices, y_indices].set(b.RENDERER.intrinsics.far)

    obs_img = b.unproject_depth_jit(depth_im, rgbd_scaled_down.intrinsics)

    return rgbd_scaled_down, obs_img, table_pose, cloud, depth_im

def add_meshes_to_renderer(
        table_dims=None # (3,)
    ):
    model_dir = os.path.join(b.utils.get_assets_dir(),"bop/ycbv/models")
    ycb_filenames = glob.glob(os.path.join(model_dir, "*.ply"))
    # The YCB objects have a standard indexing, but the order of the files in the directory
    # is not the order of the indices. So we sort the filenames by the index.
    ycb_index_order = [int(s.split("/")[-1].split("_")[-1].split(".")[0]) for s in ycb_filenames]
    sorted_ycb_filenames = [s for _,s in sorted(zip(ycb_index_order, ycb_filenames))]

    for model_path in sorted_ycb_filenames:
        b.RENDERER.add_mesh_from_file(model_path, scaling_factor=1.0/1000.0)

    # Add a cube
    cube_mesh = b.utils.make_cuboid_mesh(0.018 * jnp.ones(3))
    b.RENDERER.add_mesh(cube_mesh, "cube")

    # Add long pillars (which we can fit to the frame of the robot station)
    pillar_mesh = b.utils.make_cuboid_mesh(jnp.array([0.02, 0.02, 0.5]))
    b.RENDERER.add_mesh(pillar_mesh, "pillar")

    # Add a plane
    b.RENDERER.add_mesh_from_file("/home/georgematheos/tampura/tampura/tampura/envs/find_dice_bayes3d/environment/toy_plane.ply")

    # Optional: add table
    if table_dims is not None:
        table_mesh = b.utils.make_cuboid_mesh(table_dims)
        b.RENDERER.add_mesh(table_mesh, "table")

### Grid-based inference ###
def get_grids(param_sequence):
    return [
        b.utils.make_translation_grid_enumeration_3d(
            -x, -x, -ang, x, x, ang, *nums
        ) for (x, ang, nums) in param_sequence
    ]

def c2f(
    # n = num objects
    table_pose, # 4x4 pose
    faces, # (n,)
    potential_cps, # (n, 3)
    potential_indices, # (n,)
    number, # = n - 1
    inference_param_schedule,
    obs_img
):
    for (cp_grid, width) in inference_param_schedule:
        potential_cps, score = grid_and_max(table_pose, faces, potential_cps, potential_indices, number, cp_grid, obs_img, width)
    return potential_cps, score
c2f_jit = jax.jit(c2f)

def do_grid(
    table_pose,
    faces, # (n,)
    cps, # (n, 3)
    indices, # (n,)
    number, # = n - 1
    grid, obs_img, width
):
    cps_expanded = jnp.repeat(cps[None,...], grid.shape[0], axis=0) # (g, n, 3)
    cps_expanded = cps_expanded.at[:,number,:].set(cps_expanded[:,number,:] + grid) # (g, n, 3)
    cp_poses = cps_to_pose_parallel(cps_expanded, indices, faces, table_pose) # (g, n, 4, 4)
    rendered_images = b.RENDERER.render_many(cp_poses, indices)[...,:3] # (g, h, w, 3)
    scores = score_vmap(rendered_images, obs_img, width)
    return cps_expanded, scores

def grid_and_max(
    # n = num objects; g = num grid points
    table_pose, 
    faces, #
    cps, # (n, 3)
    indices, # (n,)
    number,
    grid,
    obs_img,
    width
):
    cps_expanded, scores = do_grid(table_pose, faces, cps, indices, number, grid, obs_img, width)
    best_idx = jnp.argmax(scores) # jnp.argsort(scores)[-4]
    cps = cps_expanded[best_idx]
    return cps, scores[best_idx]

#              (3,)
def _cp_to_pose(cp, index, face, table_pose):
    return table_pose @ b.scene_graph.relative_pose_from_edge(cp, face, b.RENDERER.model_box_dims[index])

# (n, 3) x (n,) x (n,) x (4,4) -> (n, 4, 4)
cps_to_pose = jax.vmap(_cp_to_pose, in_axes=(0,0,0,None))

# (g, n, 3) x (n,) x (n,) x (4,4) -> (g, n, 4, 4)
cps_to_pose_parallel = jax.vmap(cps_to_pose, in_axes=(0,None,None,None))
cps_to_pose_parallel_jit = jax.jit(cps_to_pose_parallel)

def score_images(
    rendered, # (h, w, 3) - point cloud
    observed, # (h, w, 3) - point cloud
    width
):
    # get L2 distance between each corresponding point
    distances = jnp.linalg.norm(observed - rendered, axis=-1)

    # Contribute 1/(h*w) * 1/width to the score for ach nearby pixel,
    # and contribute nothing for each faraway pixel.
    vals = (distances < width/2) / width
    return vals.mean()

score_vmap = jax.jit(jax.vmap(score_images, in_axes=(0, None, None)))

### Full inference loop
def run_inference(table_pose, obs_img,
        grid_param_sequence, grid_center,
        categories,
        width_sequence = None,
        key = jax.random.PRNGKey(30),
        cps = jnp.zeros((0,3)),
        indices = jnp.array([], dtype=jnp.int32),
        faces = jnp.array([], dtype=jnp.int32),
        n_objs_to_add = 1,
        possible_faces=range(6) # face_child values to consider
):
    if width_sequence is None:
        width_sequence = [0.04 for _ in grid_param_sequence]

    low, high = grid_center + jnp.array([-0.1, -0.1, -jnp.pi]), grid_center + jnp.array([0.1, 0.1, jnp.pi])

    # Expand indices to have n_objs_to_add more objects, with value -1
    prev_n_objects = len(indices)
    indices = jnp.concatenate([indices, jnp.array([-1 for _ in range(n_objs_to_add)])])
    cps = jnp.concatenate([cps, jnp.zeros((n_objs_to_add, 3))], axis=0)
    faces = jnp.concatenate([faces, jnp.array([3 for _ in range(n_objs_to_add)])])
    potential_cps = cps

    # INVARIANT: At the start of each loop iteration, `cps` and `indices`
    # contains the best fit to the scene with the number of objects tried so far.
    # best_cps and best_indices are used to track the new best state of the scene
    # after adding the next object.
    for i in range(prev_n_objects, prev_n_objects + n_objs_to_add):
        print(f"Fitting object {i}...")
        best_score = -np.inf
        best_index = -1
        best_face = -1
        best_cps = None
        best_indices = None
        best_faces = None

        # NOTE: currently using same key to fit each object
        # (but different keys across object-fitting iterations)
        key = jax.random.split(key,2)[0]
        for category_name in categories:
            # print(f"trying category {category_name}")
            next_index = category_name_to_renderer_idx(category_name)

            for next_face in possible_faces:
                potential_indices = indices.at[i].set(next_index)
                potential_cps = cps.at[i].set(jax.random.uniform(key, shape=(3,),minval=low, maxval=high))
                potential_faces = faces.at[i].set(next_face)
                potential_cps, score = c2f_jit(
                    table_pose,
                    potential_faces,
                    potential_cps,
                    potential_indices,
                    i,
                    list(zip(get_grids(grid_param_sequence), width_sequence)),
                    obs_img
                )
                if score > best_score:
                    best_index = next_index
                    best_score = score
                    best_cps = potential_cps
                    best_indices = potential_indices
                    best_face = next_face
                    best_faces = potential_faces
        cps = best_cps
        indices = best_indices
        faces = best_faces

    return cps, indices, faces, cps_to_pose(cps, indices, faces, table_pose), best_score
 
def get_viable_object_positions(
    table_pose, obs_img, grid_params,
    grid_center,
    cps, indices, faces,
    object_category,
    key=jax.random.PRNGKey(31),
    width=0.04,
    min_cp_xy=None,
    max_cp_xy=None,
    target_face_child=3
):
    low, high = grid_center + jnp.array([-0.1, -0.1, -jnp.pi]), grid_center + jnp.array([0.1, 0.1, jnp.pi])
    grid = get_grids([grid_params])[0]
    next_index = category_name_to_renderer_idx(object_category)
    potential_indices = jnp.concatenate([indices, jnp.array([next_index])])
    potential_cps = jnp.concatenate([cps, jax.random.uniform(key, shape=(1,3,),minval=low, maxval=high)])
    potential_faces = jnp.concatenate([faces, jnp.array([target_face_child])])

    _, scores_without_addition = do_grid(
        table_pose, faces,
        cps, indices,
        len(indices) - 1,
        jnp.zeros((1,3)), # grid with 1 jochange point
        obs_img,
        width
    )
    score_without_addition = scores_without_addition[0]

    expanded_cps, scores = do_grid(
        table_pose,
        potential_faces,
        potential_cps,
        potential_indices,
        len(potential_indices) - 1,
        grid,
        obs_img,
        width
    )

    if min_cp_xy is not None:
        nonclipped_indices = jnp.where((expanded_cps[:, -1, :2] >= min_cp_xy).all(axis=-1))[0]
        expanded_cps = expanded_cps[nonclipped_indices]
        scores = scores[nonclipped_indices]
    if max_cp_xy is not None:
        nonclipped_indices = jnp.where((expanded_cps[:, -1, :2] <= max_cp_xy).all(axis=-1))[0]
        expanded_cps = expanded_cps[nonclipped_indices]
        scores = scores[nonclipped_indices]

    equal_cps = expanded_cps[scores == score_without_addition]
    eq_poses = cps_to_pose_parallel(equal_cps, potential_indices, potential_faces, table_pose)
    # eq_poses is N_VALID_POSES x N_OBJECTS x 4 x 4
    strictly_greater_cps = expanded_cps[scores > score_without_addition]
    strictly_greater_poses = cps_to_pose_parallel(strictly_greater_cps, potential_indices, potential_faces, table_pose)

    return (
        equal_cps[:, -1, ...],
        strictly_greater_cps[:, -1, ...],
        eq_poses[:, -1, ...],
        strictly_greater_poses[:, -1, ...],
        potential_indices
    )

def contact_params_from_poses(table_pose, poses):
    """
    This may silently fail if the poses are not poses of objects flat on the table.
    """
    X_CT = table_pose
    X_TC = b.t3d.inverse_pose(X_CT)
    X_CO = poses # object poses in camera frame
    
    # Object poses, in table frame
    # N x 4 x 4
    X_TO = X_TC @ X_CO

    x_vals = X_TO[:, 0, 3]
    y_vals = X_TO[:, 1, 3]

    # The poses should be flat on the table, so the rotation matrix should look like
    # [cos x, -sin x, 0]
    # [sin x, cos x, 0]
    # [0, 0, 1]
    rots_around_z_axis = jnp.arccos(X_TO[:, 0, 0])
    
    # I don't understand why, but it appears I need to add pi/2
    # to get the right angle.
    rots_around_z_axis = rots_around_z_axis + np.pi/2

    # Check this rotation is approximately the same as the original    
    # new_matrices = jax.vmap(b.t3d.rotation_from_axis_angle, in_axes=(None, 0))(jnp.array([0.0, 0.0, 1.0]), rots_around_z_axis)
    # assert jnp.allclose(new_matrices, X_TO[:, :3, :3])

    # (n, 3)
    contact_param_vec = jnp.stack([x_vals, y_vals, rots_around_z_axis], axis=-1)
    assert contact_param_vec.shape == (poses.shape[0], 3)

    return contact_param_vec
