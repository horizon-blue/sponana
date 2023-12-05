from manipulation.exercises.trajectories.rrt_planner.rrt_planning import RRT as BaseRRT
from manipulation.exercises.trajectories.rrt_planner.rrt_planning import (
    Problem,
    TreeNode,
)

ConfigType = tuple[float, ...]


class RRT(BaseRRT):
    def nearest(self, configuration: ConfigType) -> TreeNode:
        """
        Finds the nearest node by distance to configuration in the
        configuration space.

        This implementation overrides the typo in BaseRRT (see
        https://github.com/RussTedrake/manipulation/issues/271 for details)

        Args:
            configuration: tuple of floats representing a configuration of a
                robot
        """
        assert self.cspace.valid_configuration(configuration)

        def recur(node: TreeNode, depth: int = 0) -> tuple[TreeNode, float]:
            closest, distance = node, self.cspace.distance(node.value, configuration)
            if depth < self.max_recursion:
                for child in node.children:
                    (child_closest, child_distance) = recur(child, depth + 1)
                    if child_distance < distance:
                        closest = child_closest
                        distance = child_distance
            return closest, distance

        return recur(self.root)[0]


class RRT_tools:
    # Adapted from: https://github.com/RussTedrake/manipulation/blob/master/trajectories/exercises/rrt_planning.ipynb
    def __init__(self, problem: Problem):
        # rrt is a tree
        self.rrt_tree = RRT(TreeNode(problem.start), problem.cspace)
        problem.rrts = [self.rrt_tree]
        self.problem = problem

    def find_nearest_node_in_RRT_graph(self, q_sample: ConfigType) -> TreeNode:
        nearest_node = self.rrt_tree.nearest(q_sample)
        return nearest_node

    def sample_node_in_configuration_space(self) -> ConfigType:
        q_sample = self.problem.cspace.sample()
        return q_sample

    def calc_intermediate_qs_wo_collision(
        self, q_start: ConfigType, q_end: ConfigType
    ) -> list[ConfigType]:
        """create more samples by linear interpolation from q_start
        to q_end. Return all samples that are not in collision

        Example interpolated path:
        q_start, qa, qb, (Obstacle), qc , q_end
        returns >>> q_start, qa, qb
        """
        return self.problem.safe_path(q_start, q_end)

    def grow_rrt_tree(self, parent_node: TreeNode, q_sample: ConfigType) -> TreeNode:
        """
        add q_sample to the rrt tree as a child of the parent node
        returns the rrt tree node generated from q_sample
        """
        child_node = self.rrt_tree.add_configuration(parent_node, q_sample)
        return child_node

    def node_reaches_goal(self, node: TreeNode) -> bool:
        return node.value == self.problem.goal

    def backup_path_from_node(self, node: TreeNode) -> list[ConfigType]:
        path = [node.value]
        while node.parent is not None:
            node = node.parent
            path.append(node.value)
        path.reverse()
        return path
