import logging

import numpy as np
from pydrake.all import Context, LeafSystem, State

logger = logging.getLogger(__name__)


class FiniteStateMachine(LeafSystem):
    """Given list of where cameras are in the scene, have spot do RRT
    and go to each of these cameras in order (using the Navigator leaf system)
    , until banana is found.
    """

    def __init__(self, target_base_positions: np.ndarray, time_step: float = 0.1):
        super().__init__()
        # TODO: remove this reshape
        self._camera_pos_list = target_base_positions.reshape(-1, 3)
        self._camera_pose_ind = self.DeclareDiscreteState(1)
        self._next_camera_pose = self.DeclareDiscreteState(3)
        self._check_banana = self.DeclareDiscreteState(1)
        self._grasp_banana = self.DeclareDiscreteState(1)
        self._do_rrt = self.DeclareDiscreteState(1)

        ### INPUT PORTS
        # check if camera has been reached (some value returned from Navigator)
        self.DeclareVectorInputPort("camera_reached", 1)
        # has banana been found? This should be returned from the perception module.
        self.DeclareVectorInputPort("see_banana", 1)
        # has banana been found? This should be returned from the grasping module.
        self.DeclareVectorInputPort("has_banana", 1)

        ### OUTPUT PORTS
        # next_camera_pose to be q_goal for the navigator
        self.DeclareStateOutputPort("target_base_position", self._next_camera_pose)
        self.DeclareStateOutputPort("check_banana", self._check_banana)
        self.DeclareStateOutputPort("grasp_banana", self._grasp_banana)
        self.DeclareStateOutputPort("do_rrt", self._do_rrt)

        self._completed: bool = False

        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=time_step,
            offset_sec=0.0,
            update=self._execute_finite_state_machine,
        )
        self.DeclareInitializationDiscreteUpdateEvent(self._initialize_state)

    def get_camera_reached_input_port(self):
        return self.GetInputPort("camera_reached")

    def get_see_banana_input_port(self):
        return self.GetInputPort("see_banana")

    def get_has_banana_input_port(self):
        return self.GetInputPort("has_banana")

    def get_do_rrt_output_port(self):
        return self.GetOutputPort("do_rrt")

    def get_check_banana_output_port(self):
        return self.GetOutputPort("check_banana")

    def get_grasp_banana_output_port(self):
        return self.GetOutputPort("grasp_banana")

    def get_target_base_position_output_port(self):
        return self.GetOutputPort("target_base_position")

    def _initialize_state(self, context: Context, state: State):
        state.set_value(self._next_camera_pose, self._camera_pos_list[0])
        state.set_value(self._do_rrt, [1])
        state.set_value(self._check_banana, [0])
        state.set_value(self._grasp_banana, [0])

    def _update_do_rrt(self, context: Context):
        """
        Function to update flag to indicate to RRT/Navigator for ready for planning/movement.
        Inputs:
        - from context,
        current_cam_reached: int where 0 when camera is reached and 1 when not.

        Returns:
        - new_cam_reached: if current cam reached, switch to 0 so that navigator can plan again.
        if current cam is not reached, stay 1 so that navigator will wait.
        """
        current_cam_reached = self.get_camera_reached_input_port().Eval(context)
        current_cam_ind = int(
            context.get_discrete_state(self._camera_pose_ind).get_value()
        )
        check_banana = int(context.get_discrete_state(self._check_banana).get_value())
        logger.debug("within _update_do_rrt function check ____")
        logger.debug(f"current_cam_reached: {current_cam_reached}")
        logger.debug(f"check_banana: {check_banana}")
        do_rrt = 1
        if current_cam_reached == 0:
            logger.debug("enable rrt to reach next camera")
            do_rrt = 1
        elif current_cam_reached == 1 and check_banana == 1:
            logger.debug("second_rrt_condition")
            do_rrt = 1
        """if current_cam_reached == 0 and current_cam_ind == 0:
            if debug_messages == True:
                logger.debug("first_rrt_condition")
            do_rrt = 1"""
        return do_rrt

    def _update_camera_ind(self, context: Context, state: State):
        """Function for updating camera pose list index.
        Inputs:
        - from context:
        current_cam_reached: int where 0 when camera is reached and 1 when not.
        see_banana: int where 0 when banana is not seen and 1 when banana seen.
        has_banana: int where 0 when banana is grasped nad 1 when banana is not grasped.

        Returns: None
        If current camera is reached and no banana is seen/grasped, needs to continue to search
        next camera pose, so current_cam_ind is incremented.
        """
        current_cam_reached = self.get_camera_reached_input_port().Eval(context)
        see_banana = self.get_see_banana_input_port().Eval(context)
        has_banana = self.get_has_banana_input_port().Eval(context)
        check_banana = int(context.get_discrete_state(self._check_banana).get_value())
        current_cam_ind = int(
            context.get_discrete_state(self._camera_pose_ind).get_value()
        )
        new_cam_ind = current_cam_ind
        num_poses = len(self._camera_pos_list)
        if current_cam_reached and check_banana and not see_banana and not has_banana:
            logger.info("curent camera reached")
            if current_cam_ind <= num_poses - 1:
                new_cam_ind += 1
            # none viewpoints have bananas, so do it all again?
            else:
                new_cam_ind = 0
        state.set_value(self._camera_pose_ind, [new_cam_ind])

    def _get_camera_pose(self, context: Context):
        """
        Function to get the next camera pose for Spot to travel to in RRT
        Inputs:
        - from context:
        current_cam_reached: int where 0 when camera is reached and 1 when not.
        Returns:
        - next camera pose
        """
        current_cam_ind = int(
            context.get_discrete_state(self._camera_pose_ind).get_value()
        )
        next_camera_pose = self._camera_pos_list[current_cam_ind]
        logger.debug("within get_camera_pose function check ______")
        logger.debug(f"current_cam_ind: {current_cam_ind}")
        logger.debug(f"next_camera_pose: {next_camera_pose}")
        return next_camera_pose

    def _update_check_banana(self, context: Context):
        """Function as indicater for perception module.
        Inputs:
        - from context:
        current_cam_reached: int where 0 when camera is reached and 1 when not.
        see_banana: int where 0 when banana is not seen and 1 when banana seen.
        has_banana: int where 0 when banana is grasped nad 1 when banana is not grasped.

        Returns:
        check_banana: if current_cam is reached, and banana is not seen, and banana is not grasped,
        return 1 to call the perception module/system.
        """
        current_cam_reached = self.get_camera_reached_input_port().Eval(context)
        see_banana = self.get_see_banana_input_port().Eval(context)
        has_banana = self.get_has_banana_input_port().Eval(context)
        check_banana = 0
        if current_cam_reached == 1 and see_banana == 0 and has_banana == 0:
            check_banana = 1
        return check_banana

    def _update_grasp_banana(self, context: Context):
        """Function as indicater for perception module.
        Inputs:
        - from context:
        current_cam_reached: int where 0 when camera is reached and 1 when not.
        see_banana: int where 0 when banana is not seen and 1 when banana seen.
        has_banana: int where 0 when banana is grasped nad 1 when banana is not grasped.

        Returns:
        grasp_banana: if current_cam is reached, and banana is seen, and banana is not grasped,
        return 1 to get banana grasped system.
        """
        current_cam_reached = self.get_camera_reached_input_port().Eval(context)
        see_banana = self.get_see_banana_input_port().Eval(context)
        has_banana = self.get_has_banana_input_port().Eval(context)
        grasp_banana = 0
        if current_cam_reached == 1 and see_banana == 1 and has_banana == 0:
            grasp_banana = 1
        return grasp_banana

    def _update_completion(self, context: Context):
        has_banana = self.get_has_banana_input_port().Eval(context)
        return bool(has_banana)

    def _execute_finite_state_machine(self, context: Context, state: State):
        if self._completed:
            return
        next_camera_pose = self._get_camera_pose(context)
        state.set_value(self._next_camera_pose, next_camera_pose)
        check_do_rrt = self._update_do_rrt(context)
        state.set_value(self._do_rrt, [check_do_rrt])
        check_banana = self._update_check_banana(context)
        state.set_value(self._check_banana, [check_banana])
        grasp_banana = self._update_grasp_banana(context)
        state.set_value(self._grasp_banana, [grasp_banana])
        self._update_camera_ind(context, state)
        completed = self._update_completion(context)
        self._completed = completed

        logger.debug(
            f"next_camera_pose: {next_camera_pose}, check_do_rrt: {check_do_rrt}, "
            f"check_banana: {check_banana}, grasp_banana: {grasp_banana}, "
            f"completed: {completed}"
        )
