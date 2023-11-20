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
    def __init__(self, camera: RgbdSensor):
        super().__init__()
        self._camera = camera

        # Input ports
        self.DeclareAbstractInputPort(
            "color_image", camera.color_image_output_port().Allocate()
        )
        self.DeclareAbstractInputPort(
            "depth_image", camera.depth_image_32F_output_port().Allocate()
        )

        # Output ports
        self.DeclareAbstractOutputPort(
            "banana_pose",
            lambda: AbstractValue.Make(RigidTransform()),
            self._locate_banana,
        )
        self.DeclareVectorOutputPort("has_banana", 1, self._find_banana)

    def get_color_image_input_port(self):
        return self.get_input_port(0)

    def get_depth_image_input_port(self):
        return self.get_input_port(1)

    def _locate_banana(self, context: Context, output: AbstractValue):
        banana_pose, _ = self._find_and_locate_banana(context)
        output.set_value(banana_pose)

    def _find_banana(self, context: Context, output: BasicVector):
        _, has_banana = self._find_and_locate_banana(context)
        output[0] = has_banana

    def _plot_images(self, context: Context):
        """for debeugging"""
        color_image = self.EvalAbstractInput(context, 0).get_value()
        depth_image = self.EvalAbstractInput(context, 1).get_value()
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(color_image.data)
        plt.subplot(1, 2, 2)
        plt.imshow(depth_image.data)

    @cache
    def _find_and_locate_banana(self, context: Context) -> (RigidTransform, bool):
        # color_image = self.EvalAbstractInput(context, 0).get_value()
        # depth_image = self.EvalAbstractInput(context, 1).get_value()
        self._plot_images(context)
        banana_pose = RigidTransform()
        has_banana = not not banana_pose
        return banana_pose, has_banana
