import matplotlib.pyplot as plt
from IPython.display import clear_output
from pydrake.all import (
    AbstractValue,
    Context,
    LeafSystem,
    Meshcat,
    RgbdSensor,
    RigidTransform,
)

class FiniteStateMachine(LeafSystem):
    """Given list of where cameras are in the scene, have spot do RRT
    and go to each of these cameras in order (using the Navigator leaf system)
    , until banana is found.
    """
    def __init__(self, camera: RgbdSensor, meshcat: Meshcat, num_tables: int = 0):
        super().__init__()
        self._camera = camera
        # Input ports


        #list of camera poses for Spot to travel to in order
        self.DeclareAbstractInputPort(
            "camera_poses", AbstractValue.Make(RigidTransform())
        )
        #spot "start state for RRT". 
        self.DeclareVectorInputPort("spot_init_state", 20)

        #has banana been found? If yes, stop going to other cameras
        #or can been changed to grasped_banana
        self.DeclareVectorInputPort("has_banana", 1)
        #check if camera has been reached (some value returned from Navigator)
        self.DeclareVectorInputPort("camera_reached", 1)

        #next_camera_pose to be q_goal for the navigator
        self.DeclareVectorOutputPort("single_cam_pose", AbstractValue.Make(RigidTransform()),
                                     self._get_camera_pose)
        #next camera to go to
        self.DeclareVectorOutputPort("camera_ind", 1, self._update_camera_ind)
        # some updated list to 
        #self.DeclareVectorOutputPort("updated_camera_poses")
        self.DeclareVectorOutputPort("completed", 1)

    def get_camera_poses_input_port(self):
        return self.get_input_port(2)
    
    def get_spot_state_input_port(self):
        return self.get_input_port(3 + self._num_tables)
    
    def get_found_banana_input_port(self):
        return self.get_input_port(1)

    def get_camera_reached(self):
        return self.get_input_port(1)
    
    def _get_camera_pose(self, context: Context) -> (RigidTransform):
        camera_reached = self.EvalAbstractInput(context, 1)
        camera_ind = self.EvalAbstractInput(context, 1)
        camera_poses = self.EvalAbstractInput(context)
        if camera_reached == True:
            camera_poses = camera_poses[camera_ind + 1]

    def _update_completion(self, context: Context):
        has_banana = self.EvalAbstractIntput(context,1)
        if has_banana == True:
            completed = True