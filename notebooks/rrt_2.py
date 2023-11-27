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
    N = 20000
    Q = np.empty((N, 3))
    rng = np.random.default_rng()
    Q[0] = q_start
    print("Q_start in Q", Q)
    goal_threshold = 0.01
    goal_reached = False

    n = 1
    while n < N:
        q_sample = rng.uniform(-6, 6, (1, 3))[0]
        q_sample[2] = q_goal[2]
        #print("plant q_sample:", q_sample)
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
        
        goal_distance = np.sum((q_goal- Q[n]) ** 2)
        if goal_distance < goal_threshold:
            goal_reached = True
            break
        n += 1
    return goal_reached, n, Q


if __name__ == "__main__":
    spot_init_state = [1.00000000e+00, 1.50392176e-12, 3.15001955e+00]
    spot_boundary = [0.006394396536052227, -9.812158532440662e-05, 0.0009113792330026627]
    table0 = [0.0, 0.0, 0.19925]
    table1 = [0.0, 2.0, 0.19925]
    table2 = [0.0, -2.0, 0.19925]
    table_bound = [0.49, 0.63, 0.015]
    tables_boundaries = [table_bound, table_bound, table_bound]
    table_poses = [table0, table1, table2]
    wall4 = [0.0, -3.5, 0.445]
    wall5 = [0.0, -1.5, 0.445]
    wall6 = [0.0, 1.5, 0.445]
    wall7 = [0.0, 3.0, 0.445]
    backwall = [-1.25, -0.25, 0.445]
    backwall_bound = [0.015, 6.49, 2.63]
    front_wall = [6.25, -0.25, 0.445]
    frontwall_bound = [0.015, 6.49, 2.63]
    #2.63
    walls_side_boundaries = [2.49, 0.015, 2.63]
    wall_poses = [wall4, wall5, wall6, wall7, backwall]
    wall_bounds = [walls_side_boundaries, walls_side_boundaries, walls_side_boundaries, walls_side_boundaries, backwall_bound]
    
    
    obs_poses = [table0, table1, table2, \
                    wall4, wall5, wall6, wall7, backwall,front_wall]
    obs_boundaries = [table_bound, table_bound, table_bound,\
                         walls_side_boundaries, walls_side_boundaries, \
                            walls_side_boundaries, walls_side_boundaries, backwall_bound, frontwall_bound]
    camera_poses_W = [[0.5567144688632728, -0.003735510093671053, 0.495],
                      [-0.4067672805262673, -0.5122634135249003, 0.495],
                      [-0.35091572089593653, 0.4881919030929625, 0.495]]
    q_goal = camera_poses_W[2]

    #goal_reached, n, Q = basic_rrt(spot_init_state, q_goal, spot_boundary, table_poses, tables_boundaries)
    goal_reached, n, Q = basic_rrt(spot_init_state, q_goal, spot_boundary, obs_poses, obs_boundaries)
    print("goal_reached:", goal_reached,"number:", n)
    if goal_reached == True:
        for i in range(n):
            print("path node:",i, Q[i])

