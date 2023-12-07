from functools import cache

import matplotlib.pyplot as plt
from pydrake.all import (
    AbstractValue,
    BasicVector,
    Context,
    LeafSystem,
    RgbdSensor,
    RigidTransform,
    State,
)


class BananaSpotter(LeafSystem):
    def __init__(self, camera: RgbdSensor, num_tables: int = 0, time_step: float = 0.1):
        super().__init__()
        self._camera = camera

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

        # Output ports
        self._found_banana = self.DeclareDiscreteState(1)
        self.DeclareStateOutputPort("found_banana", self._found_banana)
        self._banana_pose = self.DeclareAbstractState(
            AbstractValue.Make(RigidTransform())
        )
        self.DeclareStateOutputPort("banana_pose", self._banana_pose)

        self.DeclareInitializationUnrestrictedUpdateEvent(self._initialize_state)
        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec=time_step,
            offset_sec=0.0,
            update=self._try_find_banana,
        )

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
        return self.GetOutputPort("banana_pose")

    def get_found_banana_output_port(self):
        return self.GetOutputPort("found_banana")

    def _initialize_state(self, context: Context, state: State):
        state.get_mutable_discrete_state().set_value(self._found_banana, [0])
        self._set_banana_pose(state, RigidTransform())

    def _set_banana_pose(self, state: State, pose: RigidTransform):
        state.get_mutable_abstract_state(self._banana_pose).set_value(pose)

    def _try_find_banana(self, context: Context, state: State):
        should_check_banana = self.get_check_banana_input_port().Eval(context)
        if not should_check_banana:
            return

        color_image = self.EvalAbstractInput(context, 0).get_value()
        depth_image = self.EvalAbstractInput(context, 1).get_value()
        # TODO: fill in the actual perception module and delete the following lines
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(color_image.data)
        plt.subplot(1, 2, 2)
        plt.imshow(depth_image.data)
        plt.show()
