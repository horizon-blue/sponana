from typing import Callable

import numpy as np
from manipulation.exercises.trajectories.rrt_planner.robot import (
    ConfigurationSpace,
    Range,
)
from manipulation.exercises.trajectories.rrt_planner.rrt_planning import Problem

from .rrt_tools import ConfigType


class SpotProblem(Problem):
    def __init__(
        self,
        q_start: np.array,
        q_goal: np.array,
        collision_checker: Callable[[ConfigType], bool],
    ):
        self._collision_checker = collision_checker

        cspace_ranges = [
            Range(low=-6, high=6),  # base_x
            Range(low=-6, high=6),  # base_y
            Range(low=-2 * np.pi, high=2 * np.pi),  # base_rz
        ]

        # 0.1 for two prismatic joints (x, y), 2 degrees for the revolute joint (rz)
        max_steps = [0.1, 0.1, np.pi / 180 * 2]

        cspace_spot = ConfigurationSpace(cspace_ranges, np.linalg.norm, max_steps)

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

    def collide(self, configuration: ConfigType) -> bool:
        collision = self._collision_checker(configuration)
        return collision
