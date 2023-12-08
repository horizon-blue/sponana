# Bayes3D interface.

import sponana.perception.bayes3d._bayes3d as _b3d
import jax.numpy as jnp
import logging
import numpy as np
import copy

logger = logging.getLogger(__name__)

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
        known_poses: list of tuples (category_name, world-frame pose in external pose representation, face index)
            for all objects whose poses are known
        possible_target_poses: list of possible world-frame poses of the target object,
            in the external pose representation
    """

    logger.debug("In bayes3d_init.")

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

    logger.debug("Got table pose.")

    # Set up Bayes3D.  Construct the bayes3d rgbd object, scale it down
    # so we have small enough point clouds to run inference on, and set up
    # the renderer.
    rgbd = get_rgbd(camera_image, external_pose_to_b3d_pose)
    logger.debug(f"Got RGBD. scaling_factor={scaling_factor}.  table_pose_in_cam_frame={X_CT}")
    rgbd_scaled_down, obs_img, table_pose_ransac, cloud, depth_im = _b3d.scale_remove_and_setup_renderer(
        rgbd, scaling_factor=scaling_factor, table_pose_in_cam_frame=X_CT)
    logger.debug("Scaled successfully.")
    logger.debug(f"table dims = {table_dims}")
    _b3d.add_meshes_to_renderer(table_dims=table_dims)

    logger.debug("Setup renderer.")

    # Visualize preprocessed observation point cloud, and the table.
    if show_meshcat:
        _b3d.b.setup_visualizer()
        _b3d.b.clear()
        _b3d.b.show_cloud("Obs without table or too-far points", obs_img.reshape(-1,3))
        _b3d.b.show_pose("table pose", table_pose)
        _b3d.b.show_trimesh("table", _b3d.b.RENDERER.meshes[_b3d.category_name_to_renderer_idx('table')])
        _b3d.b.set_pose("table", X_CW @ X_WTcenter)

    logger.debug("Setup meshcat.")

    # Get grid enumeration schedule, based on the table size.
    center_contact_params = jnp.array([0., 0., 0.])
    grid_width = max(table_width, table_length)
    grid_param_sequence = get_grid_param_sequence(grid_width)
    min_xy = jnp.array([-table_width/2, -table_length/2])
    max_xy = jnp.array([table_width/2, table_length/2])

    logger.debug("Setup grid params.")

    if n_objects > 1:
        # Fit N-1 objects
        contact_params, category_indices, contact_faces, poses_C, no_obj_score = _b3d.run_inference(
            table_pose, obs_img, grid_param_sequence,
            center_contact_params,
            # Fit all objects other than the target, plus possibly some visible pillars
            (categories + ['pillar'] if n_pillars_to_fit > 0 else categories),
            n_objs_to_add=(n_objects - 1 + n_pillars_to_fit),
            **kwargs
        )

        # If we have fit the target object, also fit the last object
        target_fit = _b3d.category_name_to_renderer_idx(target_category) in category_indices
        if target_fit:
            contact_params, category_indices, contact_faces, poses_C, no_obj_score = _b3d.run_inference(
                table_pose, obs_img, grid_param_sequence,
                center_contact_params,
                (categories + ['pillar'] if n_pillars_to_fit > 0 else categories),
                n_objs_to_add=1,
                cps=contact_params, indices=category_indices, faces=contact_faces,
                **kwargs
            )
        logger.debug("init inference complete")
    else:
        contact_params, category_indices, poses_C, no_obj_score = jnp.zeros((0,3)), jnp.zeros((0,)), jnp.zeros((0,4,4)), 0.
        target_fit = False
        logger.debug("No init inference needed.")

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

    logger.debug("Meshcat display done.")

    if not target_fit:
        # Try fitting the target object
        if n_objects > 1:
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
        if causes_improvement and on_table:
            contact_params, category_indices, contact_faces, poses_C = updated_contact_params, updated_category_indices, updated_contact_faces, updated_poses_C
    
        logger.debug("Attempted to fit target.")

        # Visualize the best fit target location (whether or not we thought it was good enough
        # to register as visible).  (If we did not register it as visible, then the best fit
        # will probably be a meaningless pose.)
        if show_meshcat:
            _b3d.b.show_trimesh("best-fit target location", _b3d.b.RENDERER.meshes[updated_category_indices[-1]])
            _b3d.b.set_pose("best-fit target location", updated_poses_C[-1])

        logger.debug("--> + showed meshcat")

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
            (grid_param_sequence[0][0], jnp.pi, (20, 20, 8)),
            center_contact_params,
            contact_params, category_indices, contact_faces,
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

        logger.debug("Computed possible target poses.")

    # Convert poses to world frame
    X_WC = external_pose_to_b3d_pose(camera_image.camera_pose)
    poses_W = [X_WC @ X_CO for X_CO in poses_C]
    poses_W_external = [b3d_pose_to_external_pose(X_WO) for X_WO in poses_W]
    
    # Map from category to world-frame pose, in the external pose representation
    known_poses = []
    for (name, pose_W_pb, face) in zip(category_names, poses_W_external, contact_faces):
        if name != 'pillar':
            known_poses.append((name, pose_W_pb, face))

    # (4x4) @ (Nx4x4) -> (Nx4x4)
    possible_target_poses_W = X_WC @ possible_target_poses_C

    logger.debug("Conducted conversions.")

    # Return external world frame poses of the known objects, and B3D world-frame 
    # possible pose list for the target object
    return known_poses, possible_target_poses_W

        # camera_image, categories, target_category, n_objects,
        # table_info, # = (CENTERED_TABLE_POSE_W, table_width, table_length, table_thickness)
        # show_meshcat=True, scaling_factor=0.5,
        # n_pillars_to_fit=0,
        # visualize_grid=False,
        # external_pose_to_b3d_pose=None,
        # b3d_pose_to_external_pose=None,
        # **kwargs

def b3d_update(
        known_poses_W, # List of tuples (category_name, world-frame pose in external pose representation, face index)
        possible_target_locations_W, # Jax Numpy array of shape P x 4 x 4
        camera_image,
        _table_pose_W,
        target_category, # String
        show_meshcat=True,
        scaling_factor=0.5,
        external_pose_to_b3d_pose=None,
        b3d_pose_to_external_pose=None
    ):
    table_pose_W = external_pose_to_b3d_pose(_table_pose_W)
    X_WT = table_pose_W
    X_WC = external_pose_to_b3d_pose(camera_image.camera_pose)
    X_CW = _b3d.b.t3d.inverse_pose(X_WC)
    X_CT = X_CW @ X_WT # Top of table in camera frame
    table_pose = X_CT

    # preprocess data
    rgbd = get_rgbd(camera_image, external_pose_to_b3d_pose)
    rgbd_scaled_down, obs_img, table_pose_ransac, cloud, depth_im = _b3d.scale_remove_and_setup_renderer(rgbd, scaling_factor=scaling_factor, table_pose_in_cam_frame=X_CT)
    _b3d.add_meshes_to_renderer()

    if show_meshcat:
        _b3d.b.clear()
        _b3d.b.show_cloud("Obs without table or too-far points", obs_img.reshape(-1,3))
        _b3d.b.show_pose("table", table_pose)

    # get object pose array
    category_names = [cat_name for cat_name, pose, face in known_poses_W]
    faces = jnp.array([face for cat_name, pose, face in known_poses_W])
    b3d_known_poses_W = jnp.stack([external_pose_to_b3d_pose(pose) for cat_name, pose, face in known_poses_W])

    X_WC = external_pose_to_b3d_pose(camera_image.camera_pose)
    X_CW = _b3d.b.t3d.inverse_pose(X_WC)
    # N x 4 x 4
    known_poses_C = X_CW @ b3d_known_poses_W

    # (n,)
    object_indices = jnp.array([_b3d.category_name_to_renderer_idx(name) for name in category_names])
    target_object_index = _b3d.category_name_to_renderer_idx(target_category)

    if show_meshcat:
        for i in range(known_poses_C.shape[0]):
            pose_C = known_poses_C[i, ...]
            _b3d.b.show_trimesh(f"{i}", _b3d.b.RENDERER.meshes[object_indices[i]])
            _b3d.b.set_pose(f"{i}", pose_C)

    #                                                         (1 x N x 4 x 4)                     (N,)
    # 1 x w x h x 3
    rendered_without_target = _b3d.b.RENDERER.render_many(jnp.expand_dims(known_poses_C, axis=0), object_indices)[..., :3]
    # (1, )
    score_without_target = _b3d.score_vmap(rendered_without_target, obs_img, 0.04)[0]

    # P x 4 x 4
    possible_target_poses_C = X_CW @ possible_target_locations_W

    # P x N x 4 x 4
    expanded = jnp.repeat(known_poses_C[None, ...], possible_target_poses_C.shape[0], axis=0)
    # P x (N+1) x 4 x 4
    possible_pose_vecs_C = jnp.concatenate([expanded, jnp.expand_dims(possible_target_poses_C, axis=1)], axis=1)

    # P x w x h x 3
    rendered_images = _b3d.b.RENDERER.render_many(possible_pose_vecs_C, jnp.append(object_indices, target_object_index))[..., :3]

    # P
    scores = _b3d.score_vmap(rendered_images, obs_img, 0.04)
    max_score = jnp.max(scores)
    
    threshold = 0.12
    if max_score > score_without_target + threshold:
        print("Target object is visible.")
        print(f"Score without any rendered objects: {score_without_target}")
        print(f"Max score with target object: {max_score}")
        idx = jnp.argmax(scores)
        target_pose_C = do_c2f_around_pose(possible_target_poses_C[idx, ...], object_indices, faces, known_poses_C, table_pose, obs_img, target_category)
        new_possible_target_poses_C = jnp.array([target_pose_C])

        if show_meshcat:
            _b3d.b.show_trimesh(f"inferred target location", _b3d.b.RENDERER.meshes[target_object_index])
            _b3d.b.set_pose(f"inferred target location", target_pose_C)

    else:
        print("Target object is not visible.")
        print(f"Score without any rendered objects: {score_without_target}")
        print(f"Max score with target object: {max_score}")
        target_pose_C = None

        # NOTE: I have not debugged this branch on data where the target object is not visible.
        # (I did check that this code seems to do something reasonable on data where the target
        # object is visible.) 
        viable_indices = jnp.where(scores > max_score - threshold)[0]
        new_possible_target_poses_C = possible_target_poses_C[viable_indices, ...]

        if show_meshcat:
            for i in range(new_possible_target_poses_C.shape[0]):
                if i % 4 == 0:
                    _b3d.b.show_trimesh(f"possible_pose_{i}", _b3d.b.RENDERER.meshes[target_object_index])
                    _b3d.b.set_pose(f"possible_pose_{i}", new_possible_target_poses_C[i, ...])
        
    # Convert poses to world frame
    target_pose_W = (X_WC @ target_pose_C) if target_pose_C is not None else None
    new_possible_target_poses_W = X_WC @ new_possible_target_poses_C
    
    new_category_name_to_pose_W = copy.deepcopy(known_poses_W)
    if target_pose_W is not None:
        new_category_name_to_pose_W.append((target_category, b3d_pose_to_external_pose(target_pose_W), 3))

    return new_category_name_to_pose_W, new_possible_target_poses_W

#                            (4x4)         (N,)       (N,)      (N,4,4)     (4x4)     (w x h x 3)
def do_c2f_around_pose(approx_pose_C, object_indices, faces, known_poses_C, table_pose, obs_img, target_category):
    all_poses_C = jnp.concatenate([known_poses_C, jnp.expand_dims(approx_pose_C, axis=0)], axis=0)
    potential_cps = _b3d.contact_params_from_poses(table_pose, all_poses_C) # (N, 3)

    object_indices_including_target = jnp.append(object_indices, _b3d.category_name_to_renderer_idx(target_category))
    number = object_indices.shape[0]
    grid_param_schedule = get_grid_param_sequence(0.3)
    inference_param_schedule = list(zip(_b3d.get_grids(grid_param_schedule), [0.04 for _ in grid_param_schedule]))
    faces_including_target = jnp.append(faces, jnp.array([3]))
    
    optimized_cps, score = _b3d.c2f_jit(
        #               (n+1,)                 (n+1, 3)          (n+1, )
        table_pose, faces_including_target, potential_cps, object_indices_including_target, number, inference_param_schedule, obs_img
    )
    target_cp = optimized_cps[-1, ...]
    target_pose_C = _b3d._cp_to_pose(target_cp, object_indices_including_target[-1], 3, table_pose)
    return target_pose_C

# TODO: We really ought to pass in new camera intrinsics here.
# Currently this will use whatever intrinsics B3D was set up with
# in a call to b3d_init or b3d_update.
def b3d_is_visible(
        _known_poses_W,
        possible_target_pose_vec_W, # Jax Numpy array of shape P x 4 x 4; world frame
        camera_pose, # external pose representation
        target_category,
        external_pose_to_b3d_pose=None,
        b3d_pose_to_external_pose=None
    ):
    logging.info("In b3d_is_visible_vectorized")

    X_WC = external_pose_to_b3d_pose(camera_pose)
    X_CW = _b3d.b.t3d.inverse_pose(X_WC)
    possible_poses_C = X_CW @ possible_target_pose_vec_W # P x 4 x 4

    category_names = [name for name, pose, face in _known_poses_W]
    object_indices = jnp.array([_b3d.category_name_to_renderer_idx(name) for name in category_names])
    known_poses_W = jnp.stack([external_pose_to_b3d_pose(pose) for name, pose, face in _known_poses_W])
    known_poses_C = X_CW @ known_poses_W # N x 4 x 4

    target_cat_idx = _b3d.category_name_to_renderer_idx(target_category)

    # The algorithm is:
    # 1. Render the target object at each possible pose.
    # 2. Render all the objects other than the target.
    # 3. Get list of (target obj depth image) - (known obj depth image)
    # 4. Any pose where the depth image is all negative is occluded by a known object.
    #       So, return False for those poses, and True for all the others.
    #       (For robustnes I will actually require >4 visible pixels for it to count.)

    # 1. Render the target object at each possible pose.
    # P x w x h x 3
    target_rendered_images = _b3d.b.RENDERER.render_many(
        jnp.expand_dims(possible_poses_C, axis=1), # P x 1 x 4 x 4
        jnp.array([target_cat_idx]) # (1,)
    )[..., :3]
    target_rendered_depthvals = target_rendered_images[..., 2] # P x w x h x 1

    # 2. Render all the objects other than the target.
    # 1 x w x h x 3
    known_rendered_image = _b3d.b.RENDERER.render_many(
        jnp.expand_dims(known_poses_C, axis=0), # 1 x N x 4 x 4
        object_indices # (N,)
    )[0, :, :, :3]
    known_rendered_depthvals = known_rendered_image[..., 2] # w x h x 1

    # 3. Get list of (target obj depth image) - (known obj depth image)
    # P x w x h x 3
    diff_depthvals = known_rendered_depthvals - target_rendered_depthvals

    # 4. Count the number of pixels in each image where the depth is positive
    # (P,)
    num_visible_pixels = jnp.sum(diff_depthvals > 0, axis=(1,2))

    # 5. threshold
    # (P,)
    is_visible = num_visible_pixels > 4

    return is_visible
