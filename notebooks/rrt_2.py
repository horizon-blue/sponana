import numpy as np

def check_collision(q_current, spot_boundary, table_poses, table_boundary):
    """q_current: robot current position, in terms of XYtheta 
    table_poses: list of possible collisions
    """
    if q_current[0] + spot_boundary[0] > q_current[0] - spot_boundary[0]:
        spot_x_min = q_current[0] - spot_boundary[0]
        spot_x_max = q_current[0] + spot_boundary[0]
    else: 
        spot_x_min = q_current[0] + spot_boundary[0]
        spot_x_max = q_current[0] - spot_boundary[0]
    if q_current[1] + spot_boundary[1] > q_current[1] - spot_boundary[1]:
        spot_y_min = q_current[1] - spot_boundary[1]
        spot_y_max = q_current[1] + spot_boundary[1]
    else:
        spot_y_min = q_current[1] + spot_boundary[1]
        spot_y_max = q_current[1] - spot_boundary[1]
    if table_poses[0] + table_boundary[0] > table_poses[0] - table_boundary[0]: 
        table_x_min = table_poses[0] - table_boundary[0]
        table_x_max = table_poses[0] + table_boundary[0]
    else:
        table_x_min = table_poses[0] + table_boundary[0]
        table_x_max = table_poses[0] - table_boundary[0]
    if table_poses[1] +  table_boundary[1] > table_poses[1] -  table_boundary[1]:
        table_y_min = table_poses[1] -  table_boundary[1]
        table_y_max = table_poses[1] +  table_boundary[1]
    else:
        table_y_min = table_poses[1] +  table_boundary[1]
        table_y_max = table_poses[1] -  table_boundary[1]

    if spot_x_min <= table_x_max and spot_x_max >= table_x_min and \
        spot_y_min <= table_y_max and spot_y_max >= table_y_min:
        return True
    else:
        return False

def check_multiple_collisions(q_current, spot_boundary, obstacle_poses, obstacle_boundaries):
    collision_flag = False
    for i in range(len(obstacle_poses)): 
        if check_collision(q_current, spot_boundary, obstacle_poses[i], obstacle_boundaries[i]) == True:
            collision_flag = True
            break
    return collision_flag

#adapted from Rus's basic RRT example
def basic_rrt(q_start, q_goal, spot_boundary, obstacle_poses, obstacle_boundaries):
    """q_start: X and Y of starting Spot position
    q_goal: X and Y of end Spot position
    """
    max_length = 6
    N = 10000
    Q = np.empty((N, 3))
    rng = np.random.default_rng()
    Q[0] = q_start
    print("Q_start in Q", Q)
    goal_threshold = 0.1
    goal_reached = False
    #start = np.empty((N, 3))
    #end = np.empty((N, 3))
    n = 0
    while n < N:
        q_sample = rng.random((1, 3))[0]
        print("plant q_sample:", q_sample)
        distance_sq = np.sum((Q[:n] - q_sample) ** 2, axis=1)
        closest = np.argmin(distance_sq)
        distance = np.sqrt(distance_sq[closest])
        if distance > max_length:
            q_sample = Q[closest] + (max_length / distance) * (
                q_sample - Q[closest]
            )

        if check_multiple_collisions(q_sample, spot_boundary, obstacle_poses, obstacle_boundaries) == True: 
            continue

        Q[n] = q_sample
        n += 1
        goal_distance = np.sum((q_goal- Q[n]) ** 2, axis=1)
        if goal_distance < goal_threshold:
            goal_reached = True
            break
    return goal_reached, n, Q


if __name__ == "__main__":
    spot_init_state = [1.00000000e+00, 1.50392176e-12, 3.15001955e+00]
    spot_boundary = [0.006394396536052227, -9.812158532440662e-05, 0.0009113792330026627]
    table0 = [0.0, 0.0, 0.19925]
    table1 = [0.0, 2.0, 0.19925]
    table2 = [0.0, -2.0, 0.19925]
    tables_boundaries = [[0.49, 0.63, 0.015], [0.49, 0.63, 0.015], [0.49, 0.63, 0.015]]
    table_poses = [table0, table1, table2]

    camera_poses_W = [[0.5567144688632728, -0.003735510093671053, 0.495],
                      [-0.4067672805262673, -0.5122634135249003, 0.495],
                      [-0.35091572089593653, 0.4881919030929625, 0.495]]
    q_goal = camera_poses_W[0]

    goal_reached, n, Q = basic_rrt(spot_init_state, q_goal, spot_boundary, table_poses, tables_boundaries)
    print("goal_reached:", goal_reached,"number:", n, "path:", Q)

