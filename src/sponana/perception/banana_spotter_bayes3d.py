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
    possible_target_poses: list = None

    def is_initialized(self):
        return self.known_poses is not None
    
### BananaSpotter ###
"""
    BananaSpotterBayes3D

Banana spotter using Bayes3D for the underlying perception.

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
        return self.GetOutputPort("baana_pose")

    def get_found_banana_output_port(self):
        return self.GetOutputPort("found_banana")
    
    def get_perception_completed_output_port(self):
        return self.GetOutputPort("perception_completed")
    
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
        logger.info(f"Currently at table {current_table_idx}.")
        return current_table_idx
    
    def _get_images(self, context: Context, state: State):
        color_image = self.get_color_image_input_port().Eval(context).data
        depth_image = self.get_depth_image_input_port().Eval(context).data
        logger.info("Got images.")
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
            logger.info(f"Bayes3D init on table {current_table_idx}")
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
            logger.info(f"Bayes3D update on table {current_table_idx}")
            (known_poses, possible_poses) = b3d.b3d_update(
                bs.known_poses, bs.possible_target_poses, camera_image, table_pose_world_frame, 'banana',
                scaling_factor=0.2,
                external_pose_to_b3d_pose=external_pose_to_b3d_pose,
                b3d_pose_to_external_pose=b3d_pose_to_external_pose
            )
        # logger.debug(f"known poses: {known_poses} | possible_pose type: {type(possible_poses)}")
        logger.info(f"--> known pose types: {[c for (c, _, _) in known_poses]} | + {len(possible_poses)} possible banana poses")
        
        new_bs = TableBeliefState(known_poses, possible_poses)

        new_belief_states = [
            old_bs if i != current_table_idx else new_bs
            for (i, old_bs) in enumerate(table_belief_states)
        ]
        self._set_table_belief_states(state, new_belief_states)

        if _has_banana(known_poses):
            logger.info("---> Setting banana found = true.")
            self._set_banana_pose(state, _get_banana_pose(known_poses))
            self._set_found_banana(state, 1)

        state.get_mutable_discrete_state().set_value(self._perception_completed, [1])
        logger.info("Set perception_completed to true")

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