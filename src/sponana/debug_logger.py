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


class DebugLogger(LeafSystem):
    """
    A utility module for getting images & system info from the simulation
    when pressing the Space button
    """

    def __init__(self, camera: RgbdSensor, meshcat: Meshcat):
        super().__init__()

        # Input ports
        self._camera = camera
        self.DeclareAbstractInputPort(
            "color_image", camera.color_image_output_port().Allocate()
        )
        self.DeclareAbstractInputPort(
            "depth_image", camera.depth_image_32F_output_port().Allocate()
        )
        self.DeclareAbstractInputPort(
            "camera_pose", AbstractValue.Make(RigidTransform())
        )

        # Create keybind for Space button
        self._meshcat = meshcat
        self._button = "Log System Info"
        meshcat.AddButton(self._button, "Space")
        print("Press Space to log system info")

        self._click_count = 0
        # Continuously check the button pressing state and log if needed
        self.DeclarePerStepPublishEvent(self._log)

    def get_color_image_input_port(self):
        return self.get_input_port(0)

    def get_depth_image_input_port(self):
        return self.get_input_port(1)

    def get_camera_pose_input_port(self):
        return self.get_input_port(2)

    def __del__(self):
        self._meshcat.DeleteButton(self._button)

    def _plot_images(self, context: Context):
        """for debeugging"""
        color_image = self.EvalAbstractInput(context, 0).get_value()
        depth_image = self.EvalAbstractInput(context, 1).get_value()
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(color_image.data)
        plt.subplot(1, 2, 2)
        plt.imshow(depth_image.data)
        plt.show()

    def _log_camera_pose(self, context: Context):
        camera_pose = self.EvalAbstractInput(context, 2)
        if not camera_pose:  # port is not connected
            return
        camera_pose = camera_pose.get_value()
        print(f"Camera pose: {camera_pose}")

    def _log(self, context: Context):
        # check if button is clicked
        click_count = self._meshcat.GetButtonClicks(self._button)
        if click_count <= self._click_count:
            return
        clear_output()
        print("Logging system info...")
        self._log_camera_pose(context)
        self._plot_images(context)
        self._click_count = click_count
