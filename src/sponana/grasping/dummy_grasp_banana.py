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
        self._has_banana = self.DeclareDiscreteState(1)

        self.DeclareVectorInputPort("do_grasp", 1)


        self.DeclareVectorOutputPort("has_banana", 1,self._get_has_banana)
        self.DeclarePeriodicDiscreteUpdateEvent(period_sec=time_step, offset_sec=0.0, update=self._execute_grasp)

    def get_do_grasp_input_port(self):
        return self.get_input_port(0)

    def _execute_grasp(self, context, state):
        do_grasp_flag = self.get_do_grasp_input_port().Eval(context)
        has_banana = 0
        if do_grasp_flag == 1:
            #doactualgrasp
            has_banana = 1
        state.set_value(self._has_banana, has_banana)


    def _get_has_banana(self, context, output):
        has_banana = self._has_banana.Eval(context)
        output.SetFromVector(has_banana)