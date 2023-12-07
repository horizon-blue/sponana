from pydrake.all import (
    AbstractValue,
    Context,
    LeafSystem,
    RgbdSensor,
    RigidTransform,
    State,
)
import numpy as np

from ..utils import plot_two_images_side_by_side


class BananaSpotter(LeafSystem):
    def __init__(
        self,
        camera: RgbdSensor,
        num_tables: int = 0,
        time_step: float = 0.1,
        plot_camera_input: bool = False,
        table_specs: list = None
    ):
        super().__init__()
        self._camera = camera
        self._plot_camera_input = plot_camera_input
        self._table_specs = table_specs

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

        self._at_end = False

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
        return self.GetOutputPort("baana_pose")

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

        # if not self._at_end:
        if self._table_specs is not None:
            table_poses = [
                self.get_table_pose_input_port(i).Eval(context) for i in range(3)
            ]
            banana_table_idx = next((i for i, spec in enumerate(self._table_specs) if spec.has_banana), None)
            banana_table_pose = table_poses[banana_table_idx]
            camera_pose = self.get_camera_pose_input_port().Eval(context)

            dist = np.linalg.norm(banana_table_pose.translation() - camera_pose.translation())

            print(f"Dist to banana table = {dist}.")

            if dist < 1.0:
                state.get_mutable_discrete_state().set_value(self._found_banana, [1])
                print("BananaSpotter::Set found banana to true.")

        color_image = self.get_color_image_input_port().Eval(context)
        depth_image = self.get_depth_image_input_port().Eval(context)
        print("BananaSpotter::Got images.")
        # TODO: fill in the actual perception module
        if self._plot_camera_input:
            plot_two_images_side_by_side(color_image.data, depth_image.data)
        print("BananaSpotter::Displayed images.")

            # self._at_end = True
            # return

        # if self._at_end and self.get_found_banana_output_port().Eval(context)[0] == 1:
        state.get_mutable_discrete_state().set_value(self._perception_completed, [1])
        print("BananaSpotter::set perception_completed to true")