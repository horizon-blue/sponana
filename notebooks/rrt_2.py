import numpy as np

def check_collision(q_current, spot_boundary, table_poses, table_boundary):
    """q_current: robot current position
    table_poses: list of possible collisions
    """
    if q_current

#adapted from Rus's basic RRT example
def basic_rrt(q_start, q_goal):
    """q_start: X and Y of starting Spot position
    q_goal: X and Y of end Spot position
    """
    N = 10000
    Q = np.empty((N, 2))
    rng = np.random.default_rng()
    Q[0] = q_start

    start = np.empty((N, 3))
    end = np.empty((N, 3))
    for n in range(1, N):
        q_sample = rng.random((1, 2))[0]
        distance_sq = np.sum((Q[:n] - q_sample) ** 2, axis=1)
        closest = np.argmin(distance_sq)
        distance = np.sqrt(distance_sq[closest])
        if distance > 0.1:
            q_sample = Q[closest] + (0.1 / distance) * (q_sample - Q[closest])
        start[n - 1] = [Q[closest, 0], 0, Q[closest, 1]]
        end[n - 1] = [q_sample[0], 0, q_sample[1]]
        Q[n] = q_sample