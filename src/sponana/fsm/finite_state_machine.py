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
    def __init__(self, meshcat: Meshcat, num_tables: int = 0):
        super().__init__()
        self._camera_pose_ind = self.DeclareDiscreteState(1)
        self._completed = self.DeclareDiscreteState(1)
        self._next_camera_pose = self.DeclareDiscreteState(3)
        self._camera_reached = self.DeclareDiscreteState(1)
        # Input ports
        #list of camera poses for Spot to travel to in order
        self.DeclareAbstractInputPort(
            "camera_poses", [AbstractValue.Make(RigidTransform())]*9)
        #spot "start state for RRT for navigator leaf system". 
        self.DeclareVectorInputPort("spot_init_state", 20)
        #check if camera has been reached (some value returned from Navigator)
        self.DeclareVectorInputPort("camera_reached", 1)

        #has banana been found? If yes, stop going to other cameras
        #or can been changed to grasped_banana
        self.DeclareVectorInputPort("see_banana", 1)

        self.DeclareVectorInputPort("has_banana", 1)
        
        ###OUTPUT PORTS

        #next_camera_pose to be q_goal for the navigator
        self.DeclareVectorOutputPort("single_cam_pose",self._next_camera_pose)
        self.DeclareVectorOutputPort("check_banana",1)

        self.DeclareVectorOutputPort("grasp_banana", 1)

        #check if camera has been reached (some value returned from Navigator)
        self.DeclareVectorInputPort("camera_reached",self._camera_reached)


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
    
    def _update_camera_reached(self, context: Context, state: State):
        #if camera reached, switch to 0 so that RRT knows it can proceed with another trajectory
        current_cam_reached = self.get_camera_reached_input_port().Eval(context)
        new_cam_reached = 1
        if current_cam_reached == 1: 
            return 0
        state.set_value(self._camera_reached, new_cam_reached)

    def _get_camera_pose(self, context: Context, state:State):
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
        current_cam_ind = int(context.get_discrete_state(self._camera_pose_ind).get_value())
        camera_pose_list = self.get_camera_poses_input_port()
        next_camera_pose = camera_pose_list[current_cam_ind]
        state.set_value(self._next_camera_pose, next_camera_pose)
        #return current_cam_ind

    def _update_completion(self, context: Context, state: State):
        has_banana = self.get_has_banana_input_port()
        completed = 0
        if has_banana == 1:
            completed = 1
        state.set_value(self._completed, completed)