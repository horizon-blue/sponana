from pydrake.all import (
    AbstractValue,
    Context,
    LeafSystem,
    RgbdSensor,
    RigidTransform,
    State,
)

from ..utils import plot_two_images_side_by_side


class BananaSpotter(LeafSystem):
    def __init__(
        self,
        camera: RgbdSensor,
        num_tables: int = 0,
        time_step: float = 0.1,
        plot_camera_input: bool = False,
    ):
        super().__init__()
        self._camera = camera
        self._plot_camera_input = plot_camera_input

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
        
        self._perception_completed = self.DeclareDiscreteState(1)
        self.DeclareStateOutputPort("perception_completed", self._perception_completed)

        self.DeclareInitializationUnrestrictedUpdateEvent(self._initialize_state)
        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec=time_step,
            offset_sec=0.0,
            update=self._try_find_banana,
        )

        # we need to compare the current checking stage with previous one to only
        # trigger banana spotter when the checking stage changes
        self._was_checking_banana = False

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
    
    def get_perception_completed_output_port(self):
        return self.GetOutputPort("perception_completed")

    def _initialize_state(self, context: Context, state: State):
        state.get_mutable_discrete_state().set_value(self._found_banana, [0])
        self._set_banana_pose(state, RigidTransform())
        state.get_mutable_discrete_state().set_value(self._perception_completed, [0])

    def _set_banana_pose(self, state: State, pose: RigidTransform):
        state.get_mutable_abstract_state(self._banana_pose).set_value(pose)

    def _should_check_banana(self, context: Context):
        check_banana = True
        if self.get_check_banana_input_port().HasValue(context):
            check_banana = bool(self.get_check_banana_input_port().Eval(context)[0])
        retval = check_banana and not self._was_checking_banana
        self._was_checking_banana = check_banana
        return retval

    def _try_find_banana(self, context: Context, state: State):
        if not self._should_check_banana(context):
            return

        color_image = self.get_color_image_input_port().Eval(context)
        depth_image = self.get_depth_image_input_port().Eval(context)
        # TODO: fill in the actual perception module
        if self._plot_camera_input:
            plot_two_images_side_by_side(color_image.data, depth_image.data)

        # perception is now complete.
        state.get_mutable_discrete_state().set_value(self._perception_completed, [1])
