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
    and go to each of these cameras in order, until banana is found.
    """
    def __init__(self, camera: RgbdSensor, meshcat: Meshcat, num_tables: int = 0):
        super().__init__()
        self._camera = camera
        # Input ports
        self.DeclareAbstractInputPort(
            "camera_poses", AbstractValue.Make(RigidTransform())
        )
        #not sure which is better, but RRT uses spot positions
        self.DeclareVectorInputPort("spot_init_state", 20)
        self.DeclareVectorInputPort("spot_init_positions", 10)

        self.DeclareVectorOutputPort("camera_reached", 1, self.camera_reached)

    def get_camera_pose_input_port(self):
        return self.get_input_port(2)
    
    def get_spot_state_input_port(self):
        return self.get_input_port(3 + self._num_tables)

    
