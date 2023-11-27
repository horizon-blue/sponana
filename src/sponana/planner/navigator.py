from typing import Optional

import numpy as np
from manipulation.meshcat_utils import AddMeshcatTriad
from pydrake.all import (
    Context,
    LeafSystem,
    Meshcat,
    RigidTransform,
    RotationMatrix,
    State,
)

from ..rrt_2 import basic_rrt, rrt_test

def rrt_planner_dummy():
    Q, Q_split_arr = rrt_test()
    return Q_split_arr


def dummmy_planner(q_start, q_goal, num_steps: int = 20) -> list:
    """A placeholder planner that simply interpolates between the current position and the target position."""
    steps = np.linspace(0, 1, num_steps)
    q_start = np.array(q_start)
    q_goal = np.array(q_goal)
    trajectory = [(q_goal - q_start) * t + q_start for t in steps]
    print("trajectory",trajectory)
    return trajectory


class Navigator(LeafSystem):
    """Invoke the planner to get a sequence of pose (i.e. the trajectory) for Spot's
    base to navigate between rooms and tables.

    If a meshcat instance is provided, we can also visualize the trajectory.
    """

    def __init__(self, time_step: float = 1.0, meshcat: Optional[Meshcat] = None):
        super().__init__()
        self._meshcat = meshcat

        # internal states & output port
        self._base_position = self.DeclareDiscreteState(3)
        self._traj_idx = self.DeclareDiscreteState(1)
        self.DeclareStateOutputPort("base_position", self._base_position)

        # Periodically update the state to move to the next position in the
        # trajectory
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=time_step, offset_sec=0.0, update=self._update
        )

        # Input ports
        self.DeclareVectorInputPort("spot_state", 20)
        self.DeclareVectorInputPort("target_position", 3)

        # kick off the planner
        self.DeclareInitializationDiscreteUpdateEvent(self._plan_trajectory)

    def get_spot_state_input_port(self):
        return self.get_input_port(0)

    def get_target_position_input_port(self):
        return self.get_input_port(1)

    def _plan_trajectory(self, context: Context, state: State):
        current_position = self.get_spot_state_input_port().Eval(context)[:3]
        # FIXME: hard code the goal for now
        # target_position = self.get_target_position_input_port().Eval(context)
        #target_position = [2.4, 1.15, 1.65]

        # Invoke the planner to get a sequence of positions
        # TODO: replace this with a real planner
        #trajectory = dummmy_planner(current_position, target_position)
        trajectory = rrt_planner_dummy()
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

    def _update(self, context: Context, state: State):
        last_idx = int(context.get_discrete_state(self._traj_idx).get_value())
        idx = last_idx + 1 if last_idx < len(self._trajectory) - 1 else last_idx

        state.set_value(self._base_position, self._trajectory[idx])
        state.set_value(self._traj_idx, [idx])
