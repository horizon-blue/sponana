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

class DummyGrasper(LeafSystem):
    def __init__(self, time_step: float = 0.1):
        super().__init__()
        self._banana_grasped = self.DeclareDiscreteState(1)

        self.DeclareVectorInputPort("do_grasp", 1)


        self.DeclareVectorOutputPort("banana_grasped", 1,self._get_banana_grasped)
        self.DeclarePeriodicDiscreteUpdateEvent(period_sec=time_step, offset_sec=0.0, update=self._execute_grasp)

    def get_do_grasp_input_port(self):
        return self.get_input_port(0)

    def _execute_grasp(self, context, state):
        do_grasp_flag = self.get_do_grasp_input_port().Eval(context)
        banana_grasped = 0
        if do_grasp_flag == 1:
            #doactualgrasp
            banana_grasped = 1
        state.set_value(self._banana_grasped, banana_grasped)


    def _get_banana_grasped(self, context, output):
        banana_grasped = self._banana_grasped.Eval(context)
        output.SetFromVector(banana_grasped)