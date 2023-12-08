import logging

import numpy as np
from pydrake.all import Context, LeafSystem, State

logger = logging.getLogger(__name__)


class FiniteStateMachine(LeafSystem):
    """Given list of where cameras are in the scene, have spot do RRT
    and go to each of these cameras in order (using the Navigator leaf system)
    , until banana is found.

    Input ports:
    - camera_reached. = 1 when the spot is at the current target camera location. else = 0.
    - see_banana. = 1 when banana has been found. = 0 otherwise.
    - perception_completed. = 1 when perception is complete. = 0 otherwise.
    - has_banana. = 1 when banana has been grasped. = 0 otherwise.

    Output ports:
    - target_base_position. Target pose for the base.
    - do_rrt.  = 1 when the spot should be moving to the target pose.
        = 0 when spot should remain stationary (even if not at the target pose).
    - check_banana. = 1 when spot should run its perception code to update the belief state.
        = 0 otherwise.
    - grasp_banana. = 1 when spot should try to grasp the banana. = 0 otherwise.

    State:
    _camera_pose_ind = index of current target camera pose
    _current_action:
        = 1 when moving to location
        = 2 when running perception
        = 3 when running grasping
        = 4 when completed
        = 5 when this has all failed.  sad!
    """

    def __init__(self, target_base_positions: np.ndarray, time_step: float = 0.1):
        super().__init__()
        # TODO: remove this reshape
        self._camera_pos_list = target_base_positions.reshape(-1, 3)
        
        ### STATE
            # self._current_action
            # = 1 when moving to location
            # = 2 when running perception
            # = 3 when running grasping
            # = 4 when completed
            # = 5 when this has all failed.  sad!
        self._current_action = self.DeclareDiscreteState(1)
            # Index of current camera pose to go to
        self._camera_pose_ind = self.DeclareDiscreteState(1)

        ### INPUT PORTS
        self.DeclareVectorInputPort("camera_reached", 1)
        self.DeclareVectorInputPort("see_banana", 1)
        self.DeclareVectorInputPort("perception_completed", 1)
        self.DeclareVectorInputPort("has_banana", 1)

        ### OUTPUT PORTS
        self.DeclareVectorOutputPort(name="target_base_position", size=3, calc=self._set_target_base_position)
        self.DeclareVectorOutputPort(name="grasp_banana", size=1, calc=self._set_do_grasp)
        self.DeclareVectorOutputPort(name="do_rrt", size=1, calc=self._set_do_rrt)
        self.DeclareVectorOutputPort(name="check_banana", size=1, calc=self._set_check_banana)

        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=time_step,
            offset_sec=0.0,
            update=self._update,
        )
        self.DeclareInitializationDiscreteUpdateEvent(self._initialize_state)

    ### State accessors ###
    def _get_current_action(self, context):
        return int(context.get_discrete_state().get_mutable_value(self._current_action)[0])
    
    def _get_camera_pose_ind(self, context):
        ind = int(context.get_discrete_state().get_mutable_value(self._camera_pose_ind)[0])
        return ind

    ### Output port setters ##

    def _set_target_base_position(self, context, output):
        i = self._get_camera_pose_ind(context)
        output.SetFromVector(self._camera_pos_list[i])

    # Navigate when current action == 1
    def _set_do_rrt(self, context, output):
        current_action = self._get_current_action(context)
        output.SetFromVector(
            [1] if current_action == 1 else [0]
        )

    # Run perception when current action == 2
    def _set_check_banana(self, context, output):
        current_action = self._get_current_action(context)
        output.SetFromVector(
            [1] if current_action == 2 else [0]
        )

    # Run grasping when current action == 3
    def _set_do_grasp(self, context, output):
        current_action =  self._get_current_action(context)
        output.SetFromVector(
            [1] if current_action == 3 else [0]
        )

    ### Initialization ###
    def _initialize_state(self, context: Context, state: State):
        state.set_value(self._camera_pose_ind, [0])
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
                logger.debug("--> Perception completed.")
                if self._banana_visible(context, state):
                    logger.debug("----> Banana visible.")
                    # Grasp the banana
                    self._set_current_action(context, state, 3)
                else:
                    logger.debug("----> Banana not visible.")
                    still_not_done = self._increment_pose_idx(context, state)
                    if not still_not_done:
                        # Went to last position and still don't see anything.  Fail!
                        self._set_current_action(context, state, 5)
                    else:
                        self._set_current_action(context, state, 1) # Go to pose
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
                    self._set_current_action(context, state, 4) # Done!
                else:
                    logger.debug("----> Banana not obtained.")
                    self._set_current_action(context, state, 5) # Fail!
            else:
                logger.debug("--> Grasp not yet completed...")
        else:
            # Success or fail, but we're done either way
            assert current_action == 4 or current_action == 5
            # Simulation is completed.
            return

    def _increment_pose_idx(self, context, state):
        """
        Returns True if there is a next pose, False if we have
        already arived at the last one.
        """
        current_i = self._get_camera_pose_ind(context)
        if current_i >= len(self._camera_pos_list) - 1:
            return False
        else:
            state.set_value(self._camera_pose_ind, [current_i + 1])
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

    ### Output port getters: ###

    def get_do_rrt_output_port(self):
        return self.GetOutputPort("do_rrt")

    def get_check_banana_output_port(self):
        return self.GetOutputPort("check_banana")

    def get_grasp_banana_output_port(self):
        return self.GetOutputPort("grasp_banana")

    def get_target_base_position_output_port(self):
        return self.GetOutputPort("target_base_position")
