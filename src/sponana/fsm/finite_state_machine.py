import matplotlib.pyplot as plt
from IPython.display import clear_output
from pydrake.all import (
    AbstractValue,
    Context,
    LeafSystem,
    Meshcat,
    RgbdSensor,
    RigidTransform,
    SceneGraph,
    RotationMatrix, 
    State
)

class FiniteStateMachine(LeafSystem):
    """Given list of where cameras are in the scene, have spot do RRT
    and go to each of these cameras in order (using the Navigator leaf system)
    , until banana is found.
    """
    def __init__(self):
        super().__init__()
        self._camera_pose_ind = self.DeclareDiscreteState(1)
        self._completed = self.DeclareDiscreteState(1)
        self._next_camera_pose = self.DeclareDiscreteState(3)
        self._camera_reached = self.DeclareDiscreteState(1)
        self._check_banana = self.DeclareDiscreteState(1)
        self._has_banana = self.DeclareDiscreteState(1)
        self._grasp_banana = self.DeclareDiscreteState(1)
        self._do_rrt = self.DeclareDiscreteState(1)
        ### Input ports
        #list of camera poses for Spot to travel to in order
        #self.DeclareAbstractInputPort("camera_poses", [AbstractValue.Make(RigidTransform())]*9)
        self.DeclareAbstractInputPort("camera_poses", [self.DeclareDiscreteState(3)]*9)
        #spot "start state for RRT for navigator leaf system". 
        #don't know if need because navigator already gets this from the station
        self.DeclareVectorInputPort("spot_init_state", 20)
        #check if camera has been reached (some value returned from Navigator)
        self.DeclareVectorInputPort("camera_reached", 1)

        #has banana been found? This should be returned from the perception module.
        self.DeclareVectorInputPort("see_banana", 1)
        #has banana been found? This should be returned from the grasping module.
        self.DeclareVectorInputPort("has_banana", 1)
        
        ###OUTPUT PORTS

        #next_camera_pose to be q_goal for the navigator
        self.DeclareVectorOutputPort("single_cam_pose",self._next_camera_pose)
        self.DeclareVectorOutputPort("check_banana",self._check_banana)

        self.DeclareVectorOutputPort("grasp_banana", self._grasp_banana)

        
        self.DeclareVectorInputPort("do_rrt",self._do_rrt)

        self.DeclareInitializationDiscreteUpdateEvent(self._execute_finite_state_machine)
    
    def get_camera_poses_input_port(self):
        return self.get_input_port(0)
    
    def get_spot_state_input_port(self):
        return self.get_input_port(1)
    
    def get_camera_reached_input_port(self):
        return self.get_input_port(2)
    
    def get_see_banana_input_port(self):
        return self.get_input_port(3)
    
    def get_has_banana_input_port(self):
        return self.get_input_port(4)    
    
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
        current_cam_ind = int(context.get_discrete_state(self._camera_pose_ind).get_value())
        check_banana = int(context.get_discrete_state(self._check_banana).get_value())
        do_rrt = 0
        #just starting, have not reached the first camera pose, do_rrt to get to the first camera
        if current_cam_reached == 0 and current_cam_ind == 1:
            do_rrt = 1
        elif current_cam_reached == 1 and check_banana == 1: 
            do_rrt = 1
        
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
        current_cam_ind = int(context.get_discrete_state(self._camera_pose_ind).get_value())
        if current_cam_reached == 1 and see_banana == 0 and has_banana == 0:
            if current_cam_ind <= 7:
                current_cam_ind += 1
            #none viewpoints have bananas, so do it all again?
            else:
                current_cam_ind = 0
        state.set_value(self._camera_pose_ind, current_cam_ind)


    def _get_camera_pose(self, context: Context):
        """
        Function to get the next camera pose for Spot to travel to in RRT
        Inputs: 
        - from context: 
        current_cam_reached: int where 0 when camera is reached and 1 when not.
        Returns: 
        - next camera pose
        """
        current_cam_ind = int(context.get_discrete_state(self._camera_pose_ind).get_value())
        camera_pose_list = self.get_camera_poses_input_port().Eval(context)
        next_camera_pose = camera_pose_list[current_cam_ind]
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
        completed = 0
        if has_banana == 1:
            completed = 1
        return completed

    def _execute_finite_state_machine(self, context: Context, state: State):
        complete_flag = int(context.get_discrete_state(self._completed).get_value())
        while complete_flag == 0:
            check_do_rrt = self._update_do_rrt(context)
            state.set_value(self._do_rrt, check_do_rrt)
            self._update_camera_ind(context, state)
            next_camera_pose = self._get_camera_pose(context)
            state.set_value(self._next_camera_pose, next_camera_pose)
            check_banana = self._update_check_banana(context)
            state.set_value(self._check_banana, check_banana)
            grasp_banana = self._update_grasp_banana(context)
            state.set_value(self._grasp_banana, grasp_banana)
            completed = self._update_completion(context)
            state.set_value(self._completed, completed)