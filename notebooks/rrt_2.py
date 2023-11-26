import numpy as np

def check_collision(q_current, spot_boundary, table_poses, table_boundary):
    """q_current: robot current position, in terms of XYtheta 
    table_poses: list of possible collisions
    """
    spot_x_bound = [q_current[0] - spot_boundary[0], q_current[0] + spot_boundary[0]]
    spot_y_bound = [q_current[1] - spot_boundary[1], q_current[1] + spot_boundary[1]]
    table_x_bound = [table_poses[0] - table_boundary[0], table_poses[0] + table_boundary[0]]
    table_y_bound = [table_poses[1] -  table_boundary[1], table_poses[1] +  table_boundary[1]]

    if (spot_x_bound[0] > table_x_bound[0]) and (spot_y_bound[0] > table_y_bound[0]) and \
        (spot_x_bound[1] < table_x_bound[1]) and (spot_y_bound[1] < table_y_bound[1]):
        return False
    else:
        return True



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
        #while not in collision
        if distance > 0.1:
            q_sample = Q[closest] + (0.1 / distance) * (q_sample - Q[closest])
        start[n - 1] = [Q[closest, 0], 0, Q[closest, 1]]
        end[n - 1] = [q_sample[0], 0, q_sample[1]]
        Q[n] = q_sample

if __name__ == "__main__":
    spot_init_state = [1.00000000e+00, 1.50392176e-12, 3.15001955e+00]
    spot_boundary = [0.006394396536052227, -9.812158532440662e-05, 0.0009113792330026627]
    table_poses = 