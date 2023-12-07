from functools import cache

import matplotlib.pyplot as plt
from pydrake.all import (
    AbstractValue,
    BasicVector,
    Context,
    LeafSystem,
    RgbdSensor,
    RigidTransform,
)


class BananaSpotter(LeafSystem):
    def __init__(self, camera: RgbdSensor, num_tables: int = 0):
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
        self.DeclareAbstractOutputPort(
            "banana_pose",
            lambda: AbstractValue.Make(RigidTransform()),
            self._locate_banana,
        )
        self.DeclareVectorOutputPort("found_banana", 1, self._find_banana)

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

    def _locate_banana(self, context: Context, output: AbstractValue):
        banana_pose, _ = self._find_and_locate_banana(context)
        output.set_value(banana_pose)

    def _find_banana(self, context: Context, output: BasicVector):
        check_banana = self.get_check_banana_input_port().Eval(context)
        if check_banana == 1:
            _, found_banana = self._find_and_locate_banana(context)
            output[0] = found_banana
        else:
            output[0] = 0

    @cache
    def _find_and_locate_banana(self, context: Context) -> (RigidTransform, bool):
        # color_image = self.EvalAbstractInput(context, 0).get_value()
        # depth_image = self.EvalAbstractInput(context, 1).get_value()
        check_banana = self.get_check_banana_input_port().Eval(context)
        if check_banana == 1:
            banana_pose = RigidTransform()
            found_banana = not not banana_pose
            return banana_pose, found_banana
        else:
            return [], 0
