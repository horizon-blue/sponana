from typing import Optional

import numpy as np
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.station import MakeHardwareStation, load_scenario
from pydrake.all import (
    Context,
    LeafSystem,
    Meshcat,
    MultibodyPlant,
    RigidTransform,
    RotationMatrix,
    SceneGraph,
    State,
)

from ..controller import q_nominal_arm
from ..utils import configure_parser, set_spot_positions
from .rrt import ConfigType, SpotProblem, rrt_planning

default_scenario = "package://sponana/scenes/three_rooms_with_tables.dmd.yaml"


class Navigator(LeafSystem):
    """Invoke the planner to get a sequence of pose (i.e. the trajectory) for Spot's
    base to navigate between rooms and tables.

    If a meshcat instance is provided, we can also visualize the trajectory.
    """

    def __init__(
        self,
        time_step: float = 0.1,
        meshcat: Optional[Meshcat] = None,
        scenario_file: str = default_scenario,
    ):
        super().__init__()
        self._meshcat = meshcat

        # internal states & output port
        self._base_position = self.DeclareDiscreteState(3)
        self._traj_idx = self.DeclareDiscreteState(1)
        self._done_rrt = self.DeclareDiscreteState(1)
        self.DeclareStateOutputPort("base_position", self._base_position)

        # Periodically update the state to move to the next position in the
        # trajectory
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=time_step, offset_sec=2, update=self._update
        )

        # Input ports
        self.DeclareVectorInputPort("spot_state", 20)
        self.DeclareVectorInputPort("target_position", 3)
        self.DeclareVectorInputPort("do_rrt", 1)

        # Initialize internal simulation model
        self._init_internal_model(scenario_file)

        # output port for when Navigator is done
        self.DeclareVectorOutputPort(
            "done_rrt",
            1,
            self._get_done_rrt,
            prerequisites_of_calc=set([self.xd_ticket()]),
        )

        # kick off the planner
        self.DeclareInitializationDiscreteUpdateEvent(self._plan_trajectory)

    def get_spot_state_input_port(self):
        return self.get_input_port(0)

    def get_target_position_input_port(self):
        return self.get_input_port(1)

    def get_do_rrt_input_port(self):
        return self.get_input_port(2)

    def get_base_position_output_port(self):
        return self.get_output_port(0)

    def get_done_rrt_output_port(self):
        return self.get_output_port(1)

    def _plan_trajectory(self, context: Context, state: State):
        """for just moving spot to a q_sample position for collision checks in RRT"""
        do_rrt = self.get_do_rrt_input_port().Eval(context)
        if do_rrt == 1:
            current_position = self.get_spot_state_input_port().Eval(context)[:3]
            print(
                "in navigator plan trajectory: print current position:",
                current_position,
            )
            # FIXME: hard code the goal for now
            target_position = self.get_target_position_input_port().Eval(context)
            # target_position = np.array([-2, -2, 3.15001955e00]) #fixed target position test
            print(
                "in navigator plan trajectory: print target position:", target_position
            )
            spot_problem = SpotProblem(
                current_position, target_position, self._collision_check
            )
            trajectory = rrt_planning(spot_problem, max_iterations=1000)

            if self._meshcat:
                for t, pose in enumerate(trajectory):
                    # convert position to pose for plotting
                    pose = RigidTransform(
                        RotationMatrix.MakeZRotation(pose[2]), [*pose[:2], 0.0]
                    )
                    opacity = 0.2 if t > 0 and t < len(trajectory) - 1 else 1.0
                    AddMeshcatTriad(
                        self._meshcat, f"trajectory_{t}", X_PT=pose, opacity=opacity
                    )

            self._trajectory = trajectory
            # initial state
            state.set_value(self._base_position, trajectory[0])
            state.set_value(self._traj_idx, [0])
            state.set_value(self._done_rrt, [1])
        else:
            state.set_value(self._done_rrt, [0])

    def _update(self, context: Context, state: State):
        do_rrt = self.get_do_rrt_input_port().Eval(context)
        if do_rrt == 1:
            last_idx = int(context.get_discrete_state(self._traj_idx).get_value())
            idx = last_idx + 1 if last_idx < len(self._trajectory) - 1 else last_idx

            state.set_value(self._base_position, self._trajectory[idx])
            state.set_value(self._traj_idx, [idx])

    def _collision_check(self, configuration: ConfigType) -> bool:
        # move Spot to the proposed position
        spot_state = np.concatenate([configuration, q_nominal_arm])
        set_spot_positions(
            spot_state, self._station, self._station_context, visualize=False
        )
        # check for collision pairs
        return _spot_in_collision(self._plant, self._scene_graph, self._station_context)

    def _get_done_rrt(self, context, output):
        # done_rrt = self._done_rrt.Eval(context)
        done_rrt = int(context.get_discrete_state(self._done_rrt).get_value())
        output.SetFromVector([done_rrt])

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
