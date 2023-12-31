import logging
from typing import Optional

import numpy as np
from manipulation.station import load_scenario
from pydrake.all import Context, LeafSystem, Meshcat, MultibodyPlant, SceneGraph, State

from ..controller import q_nominal_arm
from ..utils import MakeSponanaHardwareStation, set_spot_positions
from .rrt import ConfigType, SpotProblem, rrt_planning
from .utils import visualize_path

default_scenario = "package://sponana/scenes/three_rooms_with_tables.dmd.yaml"
logger = logging.getLogger(__name__)


class Navigator(LeafSystem):
    """Invoke the planner to get a sequence of pose (i.e. the trajectory) for Spot's
    base to navigate between rooms and tables.

    If a meshcat instance is provided, we can also visualize the trajectory.

    Input ports:
    - spot_state
    - target_position. target spot base pose.
    - do_rrt. = 1 when the navigator should move the spot to the target_position.
        = 0 when the navigator should keep spot stationary (regardless of target_position).

    Output ports:
    - base_position. immediate next pose to move the spot to.
    - done_rrt. = 1 when spot has successfully moved to target_position.
        = 0 while in the process of moving there.
    """

    def __init__(
        self,
        time_step: float = 0.05,
        meshcat: Optional[Meshcat] = None,
        scenario_file: str = default_scenario,
        initial_position: np.ndarray = np.zeros(3),
    ):
        super().__init__()
        self._meshcat = meshcat

        # internal states & output port
        self._base_position = self.DeclareDiscreteState(3)
        self._done_rrt = self.DeclareDiscreteState(1)
        self.DeclareStateOutputPort("base_position", self._base_position)
        self.DeclareStateOutputPort("done_rrt", self._done_rrt)

        # Initialize internal simulation model
        self._init_internal_model(scenario_file)

        # Periodically update the state to move to the next position in the
        # trajectory
        self._trajectory = None
        self._traj_idx = -1
        self._previous_goal = initial_position
        self._traj_prefix = 0
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=time_step, offset_sec=0.0, update=self._update
        )

        # Input ports
        self.DeclareVectorInputPort("spot_state", 20)
        self.DeclareVectorInputPort("target_position", 3)
        self.DeclareVectorInputPort("do_rrt", 1)
        self.DeclareVectorInputPort("slow_down", 1)

        # Initialize states
        self._initial_position = initial_position
        self.DeclareInitializationDiscreteUpdateEvent(self._initialize_states)

    def get_spot_state_input_port(self):
        return self.GetInputPort("spot_state")

    def get_target_position_input_port(self):
        return self.GetInputPort("target_position")

    def get_do_rrt_input_port(self):
        return self.GetInputPort("do_rrt")

    def get_slow_down_input_port(self):
        return self.GetInputPort("slow_down")

    def get_base_position_output_port(self):
        return self.GetOutputPort("base_position")

    def get_done_rrt_output_port(self):
        return self.GetOutputPort("done_rrt")

    def _get_current_position(self, context: Context):
        return self.get_spot_state_input_port().Eval(context)[:10]

    def _initialize_states(self, context: Context, state: State):
        state.set_value(self._base_position, self._initial_position)
        state.set_value(self._done_rrt, [0])

    def _plan_trajectory(self, context: Context, state: State):
        """for just moving spot to a q_sample position for collision checks in RRT"""
        joint_positions = self._get_current_position(context)
        slow_down = self.get_slow_down_input_port().Eval(context)

        current_base_pos = joint_positions[:3]
        current_arm_pos = joint_positions[3:]
        target_position = self.get_target_position_input_port().Eval(context)
        logger.info(f"Generating path from {current_base_pos} to {target_position}")
        spot_problem = SpotProblem(
            current_base_pos,
            target_position,
            lambda config: self._collision_check(config, current_arm_pos),
            step_size=0.1 if not slow_down else 0.02,
        )
        trajectory = rrt_planning(
            spot_problem, max_n_tries=100, max_iterations_per_try=500
        )
        if self._meshcat:
            visualize_path(trajectory, self._meshcat, self._traj_prefix)
        self._traj_prefix += 1

        # reset trajectory
        self._trajectory = trajectory
        self._previous_goal = target_position
        self._traj_idx = -1

    def _should_replan(self, context: Context) -> bool:
        target_position = self.get_target_position_input_port().Eval(context)
        if self._trajectory is None:
            return True
        should_replan = not np.allclose(target_position, self._previous_goal)
        return should_replan

    def _update(self, context: Context, state: State):
        do_rrt = self.get_do_rrt_input_port().Eval(context)
        if do_rrt == 0:
            return
        if self._should_replan(context):
            self._plan_trajectory(context, state)

        if self._traj_idx < len(self._trajectory) - 1:
            self._traj_idx += 1
            state.set_value(self._base_position, self._trajectory[self._traj_idx])
            state.set_value(self._done_rrt, [0])
        else:
            state.set_value(self._done_rrt, [1])

    def _collision_check(self, configuration: ConfigType, q_arm: np.ndarray) -> bool:
        # move Spot to the proposed position
        spot_state = np.concatenate([configuration, q_arm])
        set_spot_positions(
            spot_state, self._station, self._station_context, visualize=False
        )
        # check for collision pairs
        return _spot_in_collision(self._plant, self._scene_graph, self._station_context)

    def _init_internal_model(self, scenario_file: str):
        """Initialize the planner's own internal model of the environment and use it for collision checking."""
        scenario_data = f"""
directives:
- add_directives:
    file: {scenario_file}

- add_model:
    name: spot
    file: package://sponana/spot.urdf

model_drivers:
    spot: !InverseDynamicsDriver {{}}
        """
        scenario = load_scenario(data=scenario_data)
        # Disable creation of new Meshcat instance
        scenario.visualization.enable_meshcat_creation = False
        self._station = MakeSponanaHardwareStation(scenario)
        self._plant = self._station.GetSubsystemByName("plant")
        self._scene_graph = self._station.GetSubsystemByName("scene_graph")

        self._station_context = self._station.CreateDefaultContext()


def _spot_in_collision(
    plant: MultibodyPlant, scene_graph: SceneGraph, context: Context
) -> bool:
    plant_context = plant.GetMyContextFromRoot(context)
    sg_context = scene_graph.GetMyContextFromRoot(context)
    query_object = plant.get_geometry_query_input_port().Eval(plant_context)
    inspector = scene_graph.get_query_output_port().Eval(sg_context).inspector()
    pairs = query_object.ComputePointPairPenetration()

    for pair in pairs:
        pair_name0 = inspector.GetName(pair.id_A)
        pair_name1 = inspector.GetName(pair.id_B)
        if pair_name0.startswith("spot") != pair_name1.startswith("spot"):
            return True
    return False
