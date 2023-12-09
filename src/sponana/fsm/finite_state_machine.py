import logging

import numpy as np
from pydrake.all import Context, LeafSystem, State
from pydrake.systems.framework import SystemBase

logger = logging.getLogger(__name__)


class FiniteStateMachine(LeafSystem):
    """Given list of where cameras are in the scene, have spot do RRT
    and go to each of these cameras in order (using the Navigator leaf system)
    , until banana is found.

    Constructor args:
    - target_base_positions.  IxJx3 numpy array.
        target_base_positions[table_idx, camera_idx, :]
        is the base pose for looking at table with index table_idx
        from camera_idx camera position.
        CURRENTLY J must equal 3.
    - time_step
    - is_naive_fsm.  Boolean.  True if this FSM should choose the next pose
        naively (just going in sequence).  False if the FSM should apply
        belief state reasoning to make more optimal decisions about the
        next action to take.

    Input ports:
    - camera_reached. = 1 when the spot is at the current target camera location. else = 0.
    - see_banana. = 1 when banana has been found. = 0 otherwise.
    - perception_completed. = 1 when perception is complete. = 0 otherwise.
    - has_banana. = 1 when banana has been grasped. = 0 otherwise.

    - p_pose1. Estimate of the probability that the banana is visible from pose 1
        at the current table.
    - p_pose2. Estimate of teh probability that the banana is visible from pose 2
        at the current table.

    Output ports:
    - target_base_position. Target pose for the base.
    - do_rrt.  = 1 when the spot should be moving to the target pose.
        = 0 when spot should remain stationary (even if not at the target pose).
    - check_banana. = 1 when spot should run its perception code to update the belief state.
        = 0 otherwise.
    - grasp_banana. = 1 when spot should try to grasp the banana. = 0 otherwise.

    State:
    _looking_inds:
        This specifies the (next) pose to look at the table from.
        This contains 2 numbers, (current_table_idx, current_camera_idx).
    _current_action:
        = 1 when moving to location
        = 2 when running perception
        = 3 when running grasping
        = 4 when completed
        = 5 when this has all failed.  sad!
    """

    def __init__(
        self,
        target_base_positions: np.ndarray,
        time_step: float = 0.1,
        is_naive_fsm=True,
    ):
        super().__init__()

        # 3x3x3 numpy array.
        # self._camera_pos_list[table_idx, camera_idx, :]
        # is the base pose for looking at table with index table_idx
        # from camera_idx camera position
        self._camera_pos_list = target_base_positions

        self._is_naive_fsm = is_naive_fsm

        ### STATE
        # self._current_action
        # = 1 when moving to location
        # = 2 when running perception
        # = 3 when running grasping
        # = 4 when completed
        # = 5 when this has all failed.  sad!
        self._current_action = self.DeclareDiscreteState(1)
        # Index of current camera pose to go to
        self._looking_inds = self.DeclareDiscreteState(2)

        ### INPUT PORTS
        self.DeclareVectorInputPort("camera_reached", 1)
        self.DeclareVectorInputPort("see_banana", 1)
        self.DeclareVectorInputPort("perception_completed", 1)
        self.DeclareVectorInputPort("has_banana", 1)
        self.DeclareVectorInputPort("p_pose1", 1)
        self.DeclareVectorInputPort("p_pose2", 1)

        ### OUTPUT PORTS
        self.DeclareVectorOutputPort(
            name="target_base_position", size=3, calc=self._set_target_base_position
        )
        self.DeclareVectorOutputPort(
            name="grasp_banana", size=1, calc=self._set_do_grasp
        )
        self.DeclareVectorOutputPort(name="do_rrt", size=1, calc=self._set_do_rrt)
        # Declare that `check_banana` has no direct feedthrough from the input ports.
        # (In fact, it only directly depends on _current_action, which is a state variable.)
        self.DeclareVectorOutputPort(
            "check_banana", 1, self._set_check_banana, {SystemBase.all_state_ticket()}
        )

        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=time_step,
            offset_sec=0.0,
            update=self._update,
        )
        self.DeclareInitializationDiscreteUpdateEvent(self._initialize_state)

    ### State accessors ###
    def _get_current_action(self, context):
        return int(
            context.get_discrete_state().get_mutable_value(self._current_action)[0]
        )

    def _get_looking_inds(self, context):
        vals = context.get_discrete_state().get_mutable_value(self._looking_inds)
        return (int(vals[0]), int(vals[1]))

    ### Output port setters ##

    def _set_target_base_position(self, context, output):
        (table_idx, camera_idx) = self._get_looking_inds(context)
        output.SetFromVector(self._camera_pos_list[table_idx, camera_idx, :])

    # Navigate when current action == 1
    def _set_do_rrt(self, context, output):
        current_action = self._get_current_action(context)
        output.SetFromVector([1] if current_action == 1 else [0])

    # Run perception when current action == 2
    def _set_check_banana(self, context, output):
        current_action = self._get_current_action(context)
        output.SetFromVector([1] if current_action == 2 else [0])

    # Run grasping when current action == 3
    def _set_do_grasp(self, context, output):
        current_action = self._get_current_action(context)
        output.SetFromVector([1] if current_action == 3 else [0])

    ### Initialization ###
    def _initialize_state(self, context: Context, state: State):
        state.set_value(self._looking_inds, [0, 0])
        state.set_value(self._current_action, [1])

    ### Update ###
    def _update(self, context: Context, state: State):
        current_action = self._get_current_action(context)
        if current_action == 1:
            logger.debug("Moving...")
            # Moving from point A to point B
            if self._get_rrt_completed(context, state):
                logger.debug("--> Moving completed.")
                # Run perception.
                self._set_current_action(context, state, 2)
            else:
                logger.debug("--> Moving not completed.")
            # else, continue executing the path
        elif current_action == 2:
            # Running perception
            logger.debug("Running perception...")
            if self._perception_completed(context, state):
                logger.info("--> Perception completed.")
                if self._banana_visible(context, state):
                    logger.info("----> Banana visible.")
                    # Grasp the banana
                    self._set_current_action(context, state, 3)
                else:
                    logger.info("----> Banana not visible.")
                    still_not_done = self.set_next_pose(context, state)
                    if still_not_done:
                        logger.debug("Setting FSM action to 1.")
                        self._set_current_action(context, state, 1)  # Go to pose
                        return
                    else:
                        # Went to last position and still don't see anything.  Fail!
                        self._set_current_action(context, state, 5)

            else:
                logger.debug("--> Perception not yet completed...")
            # else, continue running perception
        elif current_action == 3:
            logger.debug("Running grasping...")
            # Grasping
            if self._grasp_completed(context, state):
                logger.debug("--> Grasp completed.")
                if self._has_banana(context, state):
                    logger.debug("----> Banana obtained.")
                    self._set_current_action(context, state, 4)  # Done!
                else:
                    logger.debug("----> Banana not obtained.")
                    self._set_current_action(context, state, 5)  # Fail!
            else:
                logger.debug("--> Grasp not yet completed...")
        else:
            # Success or fail, but we're done either way
            assert current_action == 4 or current_action == 5
            # Simulation is completed.
            return

    def set_next_pose(self, context, state):
        if self._is_naive_fsm:
            return self._increment_pose_idx(context, state)
        else:
            # Do a little belief-state reasoning.
            return self._set_next_pose_using_beliefs(context, state)

    def _set_next_pose_using_beliefs(self, context, state):
        (current_table_idx, current_camera_idx) = self._get_looking_inds(context)
        (n_tables, n_camera_poses, _) = self._camera_pos_list.shape
        assert n_camera_poses == 3, "current code only supports n_camera_poses = 3"

        p_pose1 = self.get_p_pose_input_port(1).Eval(context)
        p_pose2 = self.get_p_pose_input_port(2).Eval(context)

        max_p = max(p_pose1, p_pose2)
        p1_is_max = p_pose1 == max_p

        at_last_table_and_not_done = current_table_idx == n_tables - 1 and max_p > 0
        if at_last_table_and_not_done or max_p > 0.02:
            # Go to the best remaining pose at this table.
            state.set_value(
                self._looking_inds, [current_table_idx, 1 if p1_is_max else 2]
            )
            return True
        elif current_table_idx == n_tables - 1:
            assert max_p == 0  # all tables should be explored by now
            return False  # no more exploration to do
        else:
            # Advance to first looking pose for the next table
            state.set_value(self._looking_inds, [current_table_idx + 1, 0])
            return True

    # Goes to the next camera pose in sequence.
    # (Note that this is naive behavior.)
    def _increment_pose_idx(self, context, state):
        """
        Returns True if there is a next pose, False if we have
        already arived at the last one.
        """

        logger.debug("Incrementing pose index.")

        (current_i, current_j) = self._get_looking_inds(context)

        (n_i, n_j, _) = self._camera_pos_list.shape
        logger.debug(f"n_i = {n_i}; n_j = {n_j}")
        if current_j >= n_j - 1 and current_i >= n_i - 1:
            logger.debug(
                f"At last table.  (current_j = {current_j}, current_i = {current_i})"
            )
            return False
        elif current_j >= n_j - 1:
            assert current_i + 1 < n_i
            logger.debug(
                f"incrementing table idx (current_j = {current_j}, current_i = {current_i})"
            )
            state.set_value(self._looking_inds, [current_i + 1, 0])
            return True
        else:
            assert current_j + 1 < n_j
            logger.debug(
                f"incrementing camera idx (current_j = {current_j}, current_i = {current_i})"
            )
            state.set_value(self._looking_inds, [current_i, current_j + 1])
            return True

    def _set_current_action(self, context, state, new_current_action):
        if new_current_action == 4:
            logger.info("Huge win!")
        elif new_current_action == 5:
            logger.info("Epic fail!")

        state.set_value(self._current_action, [new_current_action])

    def _get_rrt_completed(self, context, state):
        return self.get_camera_reached_input_port().Eval(context) == 1

    def _perception_completed(self, context, state):
        return self.get_perception_completed_input_port().Eval(context) == 1

    def _banana_visible(self, context, state):
        return self.get_see_banana_input_port().Eval(context) == 1

    def _grasp_completed(self, context, state):
        # TODO: have a separate input port for completion vs success
        return self.get_has_banana_input_port().Eval(context) == 1

    def _has_banana(self, context, state):
        return self.get_has_banana_input_port().Eval(context) == 1

    ### Input port getters: ###

    def get_camera_reached_input_port(self):
        return self.GetInputPort("camera_reached")

    def get_see_banana_input_port(self):
        return self.GetInputPort("see_banana")

    def get_perception_completed_input_port(self):
        return self.GetInputPort("perception_completed")

    def get_has_banana_input_port(self):
        return self.GetInputPort("has_banana")

    def get_p_pose_input_port(self, i):
        assert i == 1 or i == 2
        return self.GetInputPort(f"p_pose{i}")

    ### Output port getters: ###

    def get_do_rrt_output_port(self):
        return self.GetOutputPort("do_rrt")

    def get_check_banana_output_port(self):
        return self.GetOutputPort("check_banana")

    def get_grasp_banana_output_port(self):
        return self.GetOutputPort("grasp_banana")

    def get_target_base_position_output_port(self):
        return self.GetOutputPort("target_base_position")
