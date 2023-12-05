import numpy as np
from manipulation.exercises.trajectories.rrt_planner.rrt_planning import Problem

from .rrt_tools import ConfigType, RRT_tools


def rrt_planning(
    problem: Problem, max_iterations: int = 1000, prob_sample_q_goal: float = 0.05
) -> list[ConfigType]:
    """
    Input:
        problem (Problem): instance of a utility class
        max_iterations: the maximum number of samples to be collected
        prob_sample_q_goal: the probability of sampling q_goal

    Output:
        path (list): [q_start, ...., q_goal].

    Input: q_start, q_goal, max_interation, prob_sample_goal
    Output: path

    G.init(q_start)
    for k = 1 to max_interation:
        q_sample ← Generate Random Configuration
        random number ← random()
        if random_number < prob_sample_goal:
            q_sample ← q_goal
        n_near ← Find the nearest node in the tree(q_sample)
        (q_1, q_2, ... q_N) ← Find intermediate q's from n_near to q_sample

        // iteratively add the new nodes to the tree to form a new edge
        last_node ← n_near
        for n = 1 to N:
            last_node ← Grow RRT tree (parent_node, q_{n})

        if last node reaches the goal:
            path ← backup the path recursively
            return path

    return None"""
    rrt_tools = RRT_tools(problem)
    q_goal = problem.goal

    for _ in range(max_iterations):
        q_sample = rrt_tools.sample_node_in_configuration_space()
        if np.random.rand() < prob_sample_q_goal:
            q_sample = q_goal
        n_near = rrt_tools.find_nearest_node_in_RRT_graph(q_sample)
        intermediates = rrt_tools.calc_intermediate_qs_wo_collision(
            n_near.value, q_sample
        )

        # iteratively add the new nodes to the tree to form a new edge
        last_node = n_near
        for q_val in intermediates:
            last_node = rrt_tools.grow_rrt_tree(last_node, q_val)
            if rrt_tools.node_reaches_goal(last_node):
                return rrt_tools.backup_path_from_node(last_node)
    return []
