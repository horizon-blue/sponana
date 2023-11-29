import numpy as np
from manipulation.exercises.trajectories.rrt_planner.robot import (
    ConfigurationSpace,
    Range,
)
from manipulation.exercises.trajectories.rrt_planner.rrt_planning import Problem
from manipulation.station import MakeHardwareStation, load_scenario
from pydrake.all import Context, MultibodyPlant, SceneGraph

from ...controller import q_nominal_arm
from ...utils import configure_parser, set_spot_positions
from .rrt_tools import ConfigType

default_scenario = "package://sponana/scenes/three_rooms_with_tables.dmd.yaml"


class SpotProblem(Problem):
    def __init__(
        self, q_start: np.array, q_goal: np.array, scenario_file: str = default_scenario
    ):
        self._scenario_file = scenario_file

        cspace_ranges = [
            Range(low=-6, high=6),  # base_x
            Range(low=-6, high=6),  # base_y
            Range(low=-2 * np.pi, high=2 * np.pi),  # base_rz
        ]

        # 0.1 for two prismatic joints (x, y), 2 degrees for the revolute joint (rz)
        max_steps = [0.1, 0.1, np.pi / 180 * 2]

        cspace_spot = ConfigurationSpace(cspace_ranges, np.linalg.norm, max_steps)

        # Initialize internal simulation model
        self._init_internal_model(scenario_file)

        # Call base class constructor.
        super().__init__(
            x=10,  # not used.
            y=10,  # not used.
            robot=None,  # not used.
            obstacles=None,  # not used.
            start=tuple(q_start),
            goal=tuple(q_goal),
            cspace=cspace_spot,
        )

    def collide(self, configuration: ConfigType):
        # move Spot to the proposed position
        spot_state = np.concatenate([configuration, q_nominal_arm])
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
    file: package://manipulation/spot/spot_with_arm_and_floating_base_actuators.urdf

model_drivers:
    spot: !InverseDynamicsDriver {{}}
        """
        scenario = load_scenario(data=scenario_data)
        # Disable creation of new Meshcat instance
        scenario.visualization.enable_meshcat_creation = False
        self._station = MakeHardwareStation(
            scenario, parser_preload_callback=configure_parser
        )
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
