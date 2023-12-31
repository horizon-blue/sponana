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

from .utils import plot_two_images_side_by_side


class DebugLogger(LeafSystem):
    """
    A utility module for getting images & system info from the simulation
    when pressing the Space button
    """

    def __init__(self, camera: RgbdSensor, meshcat: Meshcat, num_tables: int = 0):
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
        self._num_tables = num_tables
        for i in range(num_tables):
            self.DeclareAbstractInputPort(
                f"table{i}_pose", AbstractValue.Make(RigidTransform())
            )
        self.DeclareVectorInputPort("spot_state", 20)
        self.DeclareAbstractInputPort(
            "banana_pose", AbstractValue.Make(RigidTransform())
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
        return self.GetInputPort("color_image")

    def get_depth_image_input_port(self):
        return self.GetInputPort("depth_image")

    def get_camera_pose_input_port(self):
        return self.GetInputPort("camera_pose")

    def get_table_pose_input_port(self, table_index: int):
        assert table_index < self._num_tables
        return self.GetInputPort(f"table{table_index}_pose")

    def get_spot_state_input_port(self):
        return self.GetInputPort("spot_state")

    def get_banana_pose_input_port(self):
        return self.GetInputPort("banana_pose")

    def __del__(self):
        self._meshcat.DeleteButton(self._button)

    def _plot_images(self, context: Context):
        """for debeugging"""
        color_image = self.get_color_image_input_port().Eval(context)
        depth_image = self.get_depth_image_input_port().Eval(context)
        plot_two_images_side_by_side(color_image.data, depth_image.data)

    def _log_camera_pose(self, context: Context):
        camera_pose = self.EvalAbstractInput(context, 2)
        if not camera_pose:  # port is not connected
            return
        camera_pose = camera_pose.get_value()
        print(f"Camera pose: {camera_pose}")

    def _log_table_pose(self, context: Context):
        for i in range(self._num_tables):
            table_pose = self.EvalAbstractInput(context, 3 + i)
            if not table_pose:  # port is not connected
                continue
            table_pose = table_pose.get_value()
            print(f"Table {i} pose: {table_pose}")

    def _log_spot_state(self, context: Context):
        spot_state = self.EvalVectorInput(context, 3 + self._num_tables)
        if spot_state is None:
            return
        spot_state = spot_state.get_value()
        # only log x, y and rotation z for now
        print(f"Spot's state: {spot_state[:3]}")

    def _log_banana_pose(self, context: Context):
        banana_pose = self.EvalAbstractInput(context, 4 + self._num_tables)
        if not banana_pose:
            return
        banana_pose = banana_pose.get_value()
        print(f"Banana pose: {banana_pose}")

    def _log(self, context: Context):
        # check if button is clicked
        click_count = self._meshcat.GetButtonClicks(self._button)
        if click_count <= self._click_count:
            return
        clear_output()
        print("Logging system info...")
        self._log_camera_pose(context)
        self._log_table_pose(context)
        self._log_spot_state(context)
        self._log_banana_pose(context)
        self._plot_images(context)
        self._click_count = click_count
