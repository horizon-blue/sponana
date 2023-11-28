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
#/home/rarama/Documents/research/sponana/notebooks/rrt_3.ipynb
#/home/rarama/Documents/research/sponana/src/sponana/planner/navigator.py

def rrt_planner_dummy():
    Q, Q_split_arr = rrt_test()
    return Q_split_arr


def interpolate_positions(q_start, q_goal, num_steps: int = 20) -> list:
    """A placeholder planner that simply interpolates between the current position and the target position."""
    steps = np.linspace(0, 1, num_steps)
    q_start = np.array(q_start)
    q_goal = np.array(q_goal)
    trajectory = (q_goal - q_start) * steps[:, None] + q_start
    return trajectory

def check_collision_move_spot(q0, q1):
    #q0 and q1 are lists of spot xytheta
    #rrt_output = [(q0[0],q0[1], q0[2]), (q1[0],q1[1],q1[2])]
    rrt_output = [
            (1.0, 1.50392176e-12, 3.15001955),
            (0.20894849, -0.47792893, 0.2475),
        ]
    return rrt_output



def dummmy_planner(*args, **kwargs):
    rrt_output = [
        (1.0, 1.50392176e-12, 3.15001955),
        (0.907556839855539, -0.7559660954414523, 3.1669330562841598),
        (0.20894849, -0.47792893, 0.2475),
    ]
    trajectory = []
    # interpolate between RRT keypoints to get a smoother trajectory
    for q_start, q_goal in zip(rrt_output[:-1], rrt_output[1:]):
        # the interpolation output contains the end points, so here we remove
        # the last point to avoid duplicates
        trajectory.extend(interpolate_positions(q_start, q_goal)[:-1])
    trajectory.append(rrt_output[-1])
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
    """
    def _execute_trajectory(self, context: Context, state: State):
        #for executing the trajectory calculated after RRT
        current_position = self.get_spot_state_input_port().Eval(context)[:3]
        # FIXME: hard code the goal for now
        # target_position = self.get_target_position_input_port().Eval(context)
        # target_position = [2.4, 1.15, 1.65]

        # Invoke the planner to get a sequence of positions
        # TODO: replace this with a real planner
        # trajectory = dummmy_planner(current_position, target_position)
        trajectory = dummmy_planner()
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
        """
    def _plan_trajectory(self, context: Context, state: State):
        """for just moving spot to a q_sample position for collision checks in RRT"""
        current_position = self.get_spot_state_input_port().Eval(context)[:3]
        # FIXME: hard code the goal for now
        # target_position = self.get_target_position_input_port().Eval(context)
        # target_position = [2.4, 1.15, 1.65]

        trajectory = check_collision_move_spot()
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
