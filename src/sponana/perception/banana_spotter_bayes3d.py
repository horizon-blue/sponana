from dataclasses import dataclass
from pydrake.all import (
    AbstractValue,
    Context,
    LeafSystem,
    RgbdSensor,
    RigidTransform,
    State,
)
import numpy as np
import sponana.perception.bayes3d.bayes3d_interface as b3d
from manipulation.scenarios import ycb

from .utils import b3d_banana_pose_to_drake
from ..utils import plot_two_images_side_by_side

from typing import NamedTuple
import jax.numpy as jnp

import logging

logger = logging.getLogger(__name__)

### Wrappers for interface to B3D ###
class CameraImage(NamedTuple):
    camera_pose: RigidTransform
    intrinsics: np.ndarray # 3 x 3
    color_image: np.ndarray # W x H x 3
    depth_image: np.ndarray # W x H

def external_pose_to_b3d_pose(x):
    return jnp.array(x.GetAsMatrix4())
def b3d_pose_to_external_pose(x):
    return RigidTransform(x)

### Belief state representation ###
@dataclass
class TableBeliefState:
    """
    Belief state about the contents of a single table.
    """
    known_poses: list = None
    possible_target_poses: np.ndarray = None

    # Mask of True/False, for each possible target pose,
    # saying whether it would be visible at each pose
    pose1_visibility_mask: np.ndarray = None
    pose2_visibility_mask: np.ndarray = None

    def is_initialized(self):
        return self.known_poses is not None
    
    def has_banana(self):
        return self.is_initialized() and _has_banana(self.known_poses)

    def n_possible_target_poses(self):
        if self.possible_target_poses is None:
            # If we haven't looked at this table,
            # all the poses are possible.
            # So to keep everything on the right scale,
            # check the number of possible poses when
            # nothing is ruled out.
            return b3d.get_max_n_possible_poses()
        else:
            return self.possible_target_poses.shape[0]

    def n_visible_at_pose(self, pose_idx):
        if pose_idx == 1:
            logger.debug(f"pose1_visibility_mask = {self.pose1_visibility_mask}")
            logger.debug(f"sum(pose1_visibility_mask) = {sum(self.pose1_visibility_mask)}")
            return sum(self.pose1_visibility_mask)
        elif pose_idx == 2:
            logger.debug(f"pose2_visibility_mask = {self.pose2_visibility_mask}")
            logger.debug(f"sum(pose2_visibility_mask) = {sum(self.pose2_visibility_mask)}")
            return sum(self.pose2_visibility_mask)
        else:
            raise AssertionError("Invalid pose idx.")
    
def get_p_visible_at_pose(table_belief_states, table_idx, camera_idx):
    assert camera_idx == 1 or camera_idx == 2

    if any([bs.has_banana() for bs in table_belief_states]):
        # If we know exactly where the banana is, return 0 or 1.
        bs = table_belief_states[table_idx]
        if bs.has_banana():
            vis_mask = bs.pose1_visibility_mask if camera_idx == 1 else bs.pose2_visibility_mask
            assert len(vis_mask) == 1, "Should only have 1 possibility at this point."
            if any(vis_mask):
                logger.info("p_pose[{table_idx}, {camera_idx}] = 1 [Known Pose]")
                return 1.
        logger.info("p_pose[{table_idx}, {camera_idx}] = 0 [Known Pose]")
        return 0.

    # Else, we have uncertainty about the banana pose and need to do some probability
    # calculations.

    logger.debug(f"evaluating p_pose[{table_idx}, {camera_idx}]")

    # P(banana on this table)
    possibility_counts = [bs.n_possible_target_poses() for bs in table_belief_states]
    p_at_table = possibility_counts[table_idx] / sum(possibility_counts)
    logger.debug(f"--> p_at_table = {p_at_table}")
    if p_at_table == 0:
        return 0

    # P(banana visible at pose 1 or pose 2 | banana on this table)
    bs = table_belief_states[table_idx]
    n_visible_pose1 = bs.n_visible_at_pose(1)
    n_visible_pose2 = bs.n_visible_at_pose(2)
    n_visible = n_visible_pose1 + n_visible_pose2
    p_visible_if_at_table = n_visible/bs.n_possible_target_poses()
    logger.debug(f"--> p_visible_if_at_table = {p_visible_if_at_table}")
    if p_visible_if_at_table == 0:
        return 0
    
    # P(banana visible at pose X | banana visible at pose 1 or pose 2)
    n_visible_at_idx = n_visible_pose1 if camera_idx == 1 else n_visible_pose2
    p_visible_at_idx_if_visible = n_visible_at_idx / n_visible
    logger.debug(f"--> p_visible_at_idx_if_visible = {p_visible_at_idx_if_visible}")

    # P(banana visible at pose X)
    p_visible_at_idx = p_at_table * p_visible_if_at_table * p_visible_at_idx_if_visible
    
    logger.info(f"p_pose[{table_idx}, {camera_idx}] = {p_visible_at_idx}")

    return p_visible_at_idx

### BananaSpotter ###
"""
    BananaSpotterBayes3D

Banana spotter using Bayes3D for the underlying perception.

Required constructor args:
- camera. RgbdSensor object from spot.
- camera_poses. 3x3 list. camera_poses[table_idx][cam_idx]
    is the (cam_idx)th camera pose at the (table_idx)th table
    (represented as a Drake RigidTransform).
- table_specs - required if n_tables > 0.

Input Ports:
- check_banana
- color_image
- depth_image
- camera_pose
- table0_pose
- table1_pose
- table2_pose

Output Ports:
- found_banana. 1 if banana found, else 0.
- banana_pose. Pose of banana (if banana has been found.)
- p_pose1.  Estimate of probability of seeing banana at pose 1 at this table.
- p_pose2.  Estimate of probability of seeing banana at pose 2 at this table.

TODO: add output ports giving the current belief about the
probability of seeing the banana at each of the other positions
Spot knows to look at this table.

State:
- _found_banana
- _table_belief_states. A list of TableBeliefState objects, describing
    the current belief about each table.
"""
class BananaSpotterBayes3D(LeafSystem):
    def __init__(
        self,
        camera: RgbdSensor,
        camera_poses: np.ndarray,
        num_tables: int = 0,
        time_step: float = 0.1,
        plot_camera_input: bool = False,
        table_specs: list = []
    ):
        super().__init__()

        assert len(table_specs) == num_tables, "Make sure to pass in table_specs"

        self._camera = camera
        self._plot_camera_input = plot_camera_input
        self._table_specs = table_specs
        self._camera_poses = camera_poses

        # Input ports
        self.DeclareVectorInputPort("check_banana", 1)
        self.DeclareAbstractInputPort(
            "color_image", camera.color_image_output_port().Allocate()
        )
        self.DeclareAbstractInputPort(
            "depth_image", camera.depth_image_32F_output_port().Allocate()
        )
        self.DeclareAbstractInputPort(
            "camera_pose", AbstractValue.Make(RigidTransform())
        )
        self._num_tables = num_tables
        for i in range(num_tables):
            self.DeclareAbstractInputPort(
                f"table{i}_pose", AbstractValue.Make(RigidTransform())
            )

        # State
        self._banana_pose = self.DeclareAbstractState(
            AbstractValue.Make(RigidTransform())
        )
        self._table_belief_states = self.DeclareAbstractState(
            AbstractValue.Make([TableBeliefState() for _ in range(num_tables)])
        )
        self._perception_completed = self.DeclareDiscreteState(1)
        self._found_banana = self.DeclareDiscreteState(1)

        # Output ports
        self.DeclareStateOutputPort("found_banana", self._found_banana)
        self.DeclareStateOutputPort("banana_pose", self._banana_pose)
        self.DeclareStateOutputPort("perception_completed", self._perception_completed)
        self.DeclareVectorOutputPort("p_pose1", 1, calc=self._set_p_pose1)
        self.DeclareVectorOutputPort("p_pose2", 1, calc=self._set_p_pose2)

        # Init & Update
        self.DeclareInitializationUnrestrictedUpdateEvent(self._initialize_state)
        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec=time_step,
            offset_sec=0.0,
            update=self._try_find_banana,
        )

        # we need to compare the current checking stage with previous one to only
        # trigger banana spotter when the checking stage changes
        self._was_checking_banana = False

    ### Port getters ###

    def get_check_banana_input_port(self):
        return self.GetInputPort("check_banana")

    def get_color_image_input_port(self):
        return self.GetInputPort("color_image")

    def get_depth_image_input_port(self):
        return self.GetInputPort("depth_image")

    def get_camera_pose_input_port(self):
        return self.GetInputPort("camera_pose")

    def get_table_pose_input_port(self, table_index: int):
        return self.GetInputPort(f"table{table_index}_pose")

    def get_banana_pose_output_port(self):
        return self.GetOutputPort("banana_pose")

    def get_found_banana_output_port(self):
        return self.GetOutputPort("found_banana")
    
    def get_perception_completed_output_port(self):
        return self.GetOutputPort("perception_completed")
    
    def get_p_pose_output_port(self, i):
        return self.GetOutputPort(f"p_pose{i}")
    
    ### Output port setters ###
    def _set_p_pose(self, context, output, i):
        belief = self._get_table_belief_states(context.get_state())
        table_idx = self._get_current_table_idx(context, context.get_state())
        p = get_p_visible_at_pose(belief, table_idx, i)
        output.SetFromVector([p])

    def _set_p_pose1(self, context, output):
        return self._set_p_pose(context, output, 1)
    
    def _set_p_pose2(self, context, output):
        return self._set_p_pose(context, output, 2)
    
    ### State getters and setters ###

    def _get_table_belief_states(self, state):
        return state.get_abstract_state(self._table_belief_states).get_value()

    def _set_table_belief_states(self, state, new_beliefs):
        state.get_mutable_abstract_state(self._table_belief_states).set_value(new_beliefs)

    def _set_banana_pose(self, state: State, pose: RigidTransform):
        state.get_mutable_abstract_state(self._banana_pose).set_value(pose)

    def _set_found_banana(self, state: State, val):
        state.get_mutable_discrete_state().set_value(self._found_banana, [val])

    ### Misc utils ###

    def _get_intrinsics(self):
        camera_info = self._camera.depth_camera_info()
        return camera_info.intrinsic_matrix()

    def _should_check_banana(self, context: Context):
        check_banana = True
        if self.get_check_banana_input_port().HasValue(context):
            check_banana = bool(self.get_check_banana_input_port().Eval(context)[0])
        retval = check_banana and not self._was_checking_banana
        self._was_checking_banana = check_banana
        return retval

    def _get_current_table_idx(self, context: Context, state: State):
        camera_pose = self.get_camera_pose_input_port().Eval(context)
        table_poses = [
            self.get_table_pose_input_port(i).Eval(context) for i in range(3)
        ]
        distances = [
            np.linalg.norm(table_pose.translation() - camera_pose.translation())
            for table_pose in table_poses
        ]
        current_table_idx = distances.index(min(distances))
        logger.debug(f"Currently at table {current_table_idx}.")
        return current_table_idx
    
    def _get_images(self, context: Context, state: State):
        color_image = self.get_color_image_input_port().Eval(context).data
        depth_image = self.get_depth_image_input_port().Eval(context).data
        logger.debug("Got images.")
        if self._plot_camera_input:
            plot_two_images_side_by_side(color_image, depth_image)
            logger.debug("Displayed images.")
        return color_image, depth_image

    ### Initialization ###
    def _initialize_state(self, context: Context, state: State):
        state.get_mutable_discrete_state().set_value(self._found_banana, [0])
        self._set_banana_pose(state, RigidTransform())
        self._set_table_belief_states(state, [TableBeliefState() for _ in self._table_specs])
        state.get_mutable_discrete_state().set_value(self._perception_completed, [0])

    ### Update ###
    def _try_find_banana(self, context: Context, state: State):
        if not self._should_check_banana(context):
            return

        current_table_idx = self._get_current_table_idx(context, state)
        color_image, depth_image = self._get_images(context, state)

        table_belief_states = self._get_table_belief_states(state)
        bs = table_belief_states[current_table_idx]
        table_spec = self._table_specs[current_table_idx]
        table_pose_world_frame = self.get_table_pose_input_port(current_table_idx).Eval(context)
        camera_image = CameraImage(
            self.get_camera_pose_input_port().Eval(context),
            self._get_intrinsics(),
            color_image[:, :, :3],
            depth_image[:, :, 0]
        )
        if not bs.is_initialized():
            logger.debug(f"Bayes3D init on table {current_table_idx}")
            (known_poses, possible_poses) = b3d.b3d_init(
                camera_image,
                _category_string_list(table_spec),
                'banana',
                table_spec.n_objects + 1, # objects + possible banana
                (table_pose_world_frame, 0.49, 0.63, 0.015),
                scaling_factor=0.2,
                external_pose_to_b3d_pose=external_pose_to_b3d_pose,
                b3d_pose_to_external_pose=b3d_pose_to_external_pose
            )
        else:
            logger.debug(f"Bayes3D update on table {current_table_idx}")
            (known_poses, possible_poses) = b3d.b3d_update(
                bs.known_poses, bs.possible_target_poses, camera_image, table_pose_world_frame, 'banana',
                scaling_factor=0.2,
                external_pose_to_b3d_pose=external_pose_to_b3d_pose,
                b3d_pose_to_external_pose=b3d_pose_to_external_pose
            )
        # logger.debug(f"known poses: {known_poses} | possible_pose type: {type(possible_poses)}")
        logger.info(f"--> known pose types: {[c for (c, _, _) in known_poses]} | + {len(possible_poses)} possible banana poses")
        
        logger.debug(f"current_table_idx = {current_table_idx}")
        logger.debug(f"self._camera_poses = {self._camera_poses}")
        logger.debug(f"Cam 1 pose: {self._camera_poses[current_table_idx][1]}")
        logger.debug(f"Cam 2 pose: {self._camera_poses[current_table_idx][2]}")
        logger.debug(f"Current camera pose: {self.get_camera_pose_input_port().Eval(context)}")
        pose1_visibility_mask = b3d.b3d_is_visible(
            known_poses, possible_poses,
            self._camera_poses[current_table_idx][1],
            'banana',
            external_pose_to_b3d_pose=external_pose_to_b3d_pose,
            b3d_pose_to_external_pose=b3d_pose_to_external_pose
        )
        logger.debug(f"--> Got pose 1 visibility mask.  Sum = {sum(pose1_visibility_mask)}")
        pose2_visibility_mask = b3d.b3d_is_visible(
            known_poses, possible_poses,
            self._camera_poses[current_table_idx][2],
            'banana',
            external_pose_to_b3d_pose=external_pose_to_b3d_pose,
            b3d_pose_to_external_pose=b3d_pose_to_external_pose
        )
        logger.debug(f"--> Got pose 2 visibility mask.  Sum = {sum(pose2_visibility_mask)}")
        new_bs = TableBeliefState(known_poses, possible_poses, pose1_visibility_mask, pose2_visibility_mask)
        logger.debug(f"--> belief state constructed.")

        new_belief_states = [
            old_bs if i != current_table_idx else new_bs
            for (i, old_bs) in enumerate(table_belief_states)
        ]
        self._set_table_belief_states(state, new_belief_states)

        if _has_banana(known_poses):
            banana_pose_b3d = _get_banana_pose(known_poses)

            # Somehow the way the poses correspond to the meshes is different in Drake
            # and Bayes3D.  This is the transform that takes the pose b3d says the banana
            # is at, and converts it to the pose Drake should think the banana is at.
            banana_pose_drake = b3d_banana_pose_to_drake(banana_pose_b3d)
            self._set_banana_pose(state, banana_pose_drake)
            self._set_found_banana(state, 1)

            logger.debug("---> Setting banana found = true.")
            logger.debug(f"---> Bayes3D banana pose: {banana_pose_b3d}")
            logger.debug(f"---> Inferred Drake banana pose: {banana_pose_drake}")

        state.get_mutable_discrete_state().set_value(self._perception_completed, [1])
        logger.debug("Set perception_completed to true")

def _has_banana(known_poses):
    for (category_name, pose, face) in known_poses:
        if category_name == "banana":
            return True
    return False

def _get_banana_pose(known_poses):
    for (category_name, pose, face) in known_poses:
        if category_name == "banana":
            return pose
    
    raise AssertionError("No banana in known poses list.")

def _category_string_list(table_spec):
    """
    Returns a list of strings (INCLUDING "banana") corresponding to all
    objects on the table, per the table_spec.
    """
    # TODO: just change the table_spec interface to directly store string names.
    # That will be less confusing anyway.

    # first 4 characters are the ycb index; strip these away
    cats = [ycb[i][4:] for i in table_spec.object_type_indices]
    cats = [cat.split(".")[0] for cat in cats]
    
    if "banana" not in cats:
        cats.append("banana")
    
    return cats