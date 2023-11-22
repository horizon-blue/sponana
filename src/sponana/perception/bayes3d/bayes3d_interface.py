# Bayes3D interface.

import sponana.perception.bayes3d._bayes3d as _b3d
import jax.numpy as jnp
import logging
import numpy as np
import copy

def xy_on_table(xyz_T, table_length, table_width):
    x = xyz_T[0]
    y = xyz_T[1]
    return (x > -table_width/2) and (x < table_width/2) and (y > -table_length/2) and (y < table_length/2)

def get_grid_param_sequence(table_width):
    return [
        (table_width / 2, jnp.pi, (20, 20, 40)),
        (0.2, jnp.pi * 3/4, (15, 15, 15)),
        (0.1, jnp.pi * 1/2, (15, 15, 15)),
        (0.05, jnp.pi * 1/3, (15,15,15)),
        (0.02, jnp.pi * 1/4, (9,9,51))
    ]


def get_rgbd(img, external_pose_to_b3d_pose):
    b = _b3d.b
    K = img.intrinsics
    fx, fy, cx, cy = K[0,0],K[1,1],K[0,2],K[1,2]
    h,w = img.depth_image.shape
    intrinsics = b.Intrinsics(h,w,fx,fy,cx,cy,0.001,10000.0)
    rgbd = b.RGBD(
        img.color_image,
        img.depth_image,
        external_pose_to_b3d_pose(img.camera_pose),
        intrinsics
    )
    return rgbd

### Main functionality below this point ###

# Conventions:
# _Pose = pose represented in the external (Drake) representation
# Pose = pose represented in bayes3d representation
# W = world frame
# C = current camera frame
# T = top of table frame
# Tcenter = center of table frame
def b3d_init(
        camera_image, categories, target_category, n_objects,
        table_info, # = (CENTERED_TABLE_POSE_W, table_width, table_length, table_thickness)
        show_meshcat=True, scaling_factor=0.5,
        n_pillars_to_fit=0,
        visualize_grid=False,
        external_pose_to_b3d_pose=None,
        b3d_pose_to_external_pose=None,
        **kwargs
    ):
    """
    Args:
        camera_image: CameraImage object to be converted to a B3D image
        categories: list of strings, names of categories for objects which may be present in the scene
        target_category: string, category name of the target object
        n_objects: int, number of objects to fit
        [show_meshcat=True]: bool, whether to show the MeshCat visualizer
        scaling_factor: float, scaling factor for the B3D image
        external_pose_to_b3d_pose: function, converts external pose representation to B3d pose representation
        b3d_pose_to_external_pose: function, converts B3d pose representation to external pose representation
    Returns:
        known_poses: list of tuples (category_name, world-frame pose in external pose representation)
            for all objects whose poses are known
        possible_target_poses: list of possible world-frame poses of the target object,
            in the external pose representation
    """

    # Get the table dimensions, and the table pose in the camera frame.
    _CENTERED_TABLE_POSE_W, table_width, table_length, table_thickness = table_info
    CENTERED_TABLE_POSE_W = external_pose_to_b3d_pose(_CENTERED_TABLE_POSE_W)
    table_dims = jnp.array([table_width, table_length, table_thickness])
    X_TcenterT = _b3d.b.t3d.transform_from_pos(jnp.array([0., 0., table_thickness/2]))
    X_WTcenter = CENTERED_TABLE_POSE_W
    X_WT = X_WTcenter @ X_TcenterT
    X_WC = external_pose_to_b3d_pose(camera_image.camera_pose)
    X_CW = _b3d.b.t3d.inverse_pose(X_WC)
    X_CT = X_CW @ X_WT # Top of table in camera frame
    table_pose = X_CT

    # Set up Bayes3D.  Construct the bayes3d rgbd object, scale it down
    # so we have small enough point clouds to run inference on, and set up
    # the renderer.
    rgbd = get_rgbd(camera_image, external_pose_to_b3d_pose)
    rgbd_scaled_down, obs_img, table_pose_ransac, cloud, depth_im = _b3d.scale_remove_and_setup_renderer(
        rgbd, scaling_factor=scaling_factor, table_pose_in_cam_frame=X_CT)
    _b3d.add_meshes_to_renderer(table_dims=table_dims)

    # Visualize preprocessed observation point cloud, and the table.
    if show_meshcat:
        _b3d.b.setup_visualizer()
        _b3d.b.clear()
        _b3d.b.show_cloud("Obs without table or too-far points", obs_img.reshape(-1,3))
        _b3d.b.show_pose("table pose", table_pose)
        _b3d.b.show_trimesh("table", _b3d.b.RENDERER.meshes[_b3d.category_name_to_renderer_idx('table')])
        _b3d.b.set_pose("table", X_CW @ X_WTcenter)

    # Get grid enumeration schedule, based on the table size.
    center_contact_params = jnp.array([0., 0., 0.])
    grid_width = max(table_width, table_length)
    grid_param_sequence = get_grid_param_sequence(grid_width)
    min_xy = jnp.array([-table_width/2, -table_length/2])
    max_xy = jnp.array([table_width/2, table_length/2])

    # Fit all objects other than the target (including the 2 visible pillars)
    nontarget_categories = [cat for cat in categories if cat != target_category]
    if len(nontarget_categories) > 0:
        contact_params, category_indices, contact_faces, poses_C, no_obj_score = _b3d.run_inference(
            table_pose, obs_img, grid_param_sequence,
            center_contact_params,
            # Fit all objects other than the target, plus possibly some visible pillars
            (nontarget_categories + ['pillar'] if n_pillars_to_fit > 0 else nontarget_categories),
            n_objs_to_add=(n_objects - 1 + n_pillars_to_fit),
            **kwargs
        )
        logging.info(f"Score without target object: {no_obj_score}")
    else:
        contact_params, category_indices, poses_C, no_obj_score = jnp.zeros((0,3)), jnp.zeros((0,)), jnp.zeros((0,4,4)), 0.

    # Visualize the fitted objects
    if show_meshcat:
        for i in range(len(poses_C)):
            _b3d.b.show_trimesh(f"{i}", _b3d.b.RENDERER.meshes[category_indices[i]])
            _b3d.b.set_pose(f"{i}", poses_C[i])

        # Optional: visualize the coarsest-level grid
        # used for inference (useful for debugging).
        if visualize_grid:
            cps = _b3d.get_grids(grid_param_sequence)[0]
            poses = _b3d.cps_to_pose_parallel(
                cps.reshape(-1, 1, 3),
                jnp.array([_b3d.category_name_to_renderer_idx(categories[0])]),
                table_pose, 3
            )
            for i in range(cps.shape[0]):
                if i % 4 == 0:
                    _b3d.b.show_pose(f"grid pose {i}", poses[i, ...])

    # Try adding the target object
    if len(nontarget_categories) > 0:
        updated_contact_params, updated_category_indices, updated_contact_faces, updated_poses_C, with_obj_score = _b3d.run_inference(
            table_pose, obs_img, grid_param_sequence, center_contact_params, [target_category],
            cps=contact_params, indices=category_indices, faces=contact_faces,
            n_objs_to_add=1, possible_faces=[3]
        )
    else:
        updated_contact_params, updated_category_indices, updated_contact_faces, updated_poses_C, with_obj_score = _b3d.run_inference(
            table_pose, obs_img, grid_param_sequence, center_contact_params, [target_category],
            n_objs_to_add=1, possible_faces=[3]
        )
    logging.info(f"Score with target object: {with_obj_score}")

    # If the score with the target object is sufficiently higher than the score without it,
    # then we register the target object as visible.
    causes_improvement = with_obj_score > no_obj_score + 0.02
    on_table = xy_on_table(jnp.array([updated_contact_params[-1, 0], updated_contact_params[-1, 1], 0.]), table_length, table_width)
    print(f"On table: {on_table}")
    if causes_improvement and on_table:
        logging.info("Target object registered as visible.")
        contact_params, category_indices, contact_faces, poses_C = updated_contact_params, updated_category_indices, updated_contact_faces, updated_poses_C
    
    # Visualize the best fit target location (whether or not we thought it was good enough
    # to register as visible).  (If we did not register it as visible, then the best fit
    # will probably be a meaningless pose.)
    if show_meshcat:
        _b3d.b.show_trimesh("best-fit target location", _b3d.b.RENDERER.meshes[updated_category_indices[-1]])
        _b3d.b.set_pose("best-fit target location", updated_poses_C[-1])

    # Get the possible poses of the target object
    category_names = [_b3d.renderer_idx_to_category_name(i) for i in category_indices]
    if target_category in category_names:
        # If we have registered the target object as visible, then we know its pose.
        logging.info(f"Target category {target_category} found in scene.")
        possible_target_poses_C = jnp.array([poses_C[category_names.index(target_category)]])
    else:
        # Otherwise, do an enumeration scan and see which poses are consistent with the observation.
        (
            equal_cps, strictly_greater_cps,
            eq_poses, strictly_greater_poses,
            potential_indices
        )  = _b3d.get_viable_object_positions(
            table_pose, obs_img,
            (grid_param_sequence[0][0], jnp.pi, (50, 50, 1)),
            center_contact_params,
            contact_params, category_indices,
            target_category,
            min_cp_xy=min_xy,
            max_cp_xy=max_xy
        )
        possible_target_poses_C = jnp.concatenate([eq_poses, strictly_greater_poses], axis=0)

        # Visualize [a subset of] the possible poses
        # (subset is just so the visualization isn't too slow)
        if show_meshcat:
            for i in range(possible_target_poses_C.shape[0]):
                if i % 4 == 0:
                    _b3d.b.show_trimesh(f"possible_pose_{i}", _b3d.b.RENDERER.meshes[potential_indices[-1]])
                    _b3d.b.set_pose(f"possible_pose_{i}", possible_target_poses_C[i, ...])

    # Convert poses to world frame
    X_WC = external_pose_to_b3d_pose(camera_image.camera_pose)
    poses_W = [X_WC @ X_CO for X_CO in poses_C]
    poses_W_external = [b3d_pose_to_external_pose(X_WO) for X_WO in poses_W]
    
    # Map from category to world-frame pose, in the external pose representation
    known_poses = []
    for (name, pose_W_pb) in zip(category_names, poses_W_external):
        if name != 'pillar':
            known_poses.append((name, pose_W_pb))

    # (4x4) @ (Nx4x4) -> (Nx4x4)
    possible_target_poses_W = X_WC @ possible_target_poses_C

    # Return external world frame poses of the known objects, and B3D world-frame 
    # possible pose list for the target object
    return known_poses, possible_target_poses_W
