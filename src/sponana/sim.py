import numpy as np
import random
import math
from pydrake.common import temp_directory
from pydrake.geometry import StartMeshcat
from pydrake.systems.analysis import Simulator
from pydrake.visualization import ModelVisualizer
from pydrake.all import Simulator, StartMeshcat, LogVectorOutput

from manipulation import running_as_notebook
from manipulation.station import MakeHardwareStation, load_scenario
from IPython.display import HTML, display
from matplotlib import pyplot as plt
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    ConnectPlanarSceneGraphVisualizer,
    DiagramBuilder,
    FixedOffsetFrame,
    JointIndex,
    Parser,
    PlanarJoint,
    RandomGenerator,
    RigidTransform,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    UniformlyRandomRotationMatrix,
)

from manipulation import ConfigureParser, running_as_notebook
from manipulation.scenarios import AddShape, ycb
from manipulation.station import MakeHardwareStation, load_scenario
from pydrake.common import temp_directory

# sponana/src/sponana/utils.py
import sponana.utils
from sponana.controller import SpotController
from sponana.debug_logger import DebugLogger
from sponana.perception import (
    add_camera_pose_extractor,
    add_body_pose_extractor,
    BananaSpotter,
)
from sponana.hardcoded_cameras import get_camera_generator_str

def distance_thres_point(poss_point_x, poss_point_y, x_points, y_points, r_threshold):
    # print("poss_point_x", poss_point_x, "poss_point_y", poss_point_y)
    for ind in range(len(x_points)):
        dist = math.hypot(
            abs(poss_point_x - x_points[ind]), abs(poss_point_y - y_points[ind])
        )
        print("dist", dist)
        if dist < r_threshold:
            return False
    return True


def generate_random_with_min_dist(x_upper_range, y_upper_range, num_elements):
    x_points = []
    y_points = []
    # max x collisions of the ycb objects: 0.158000 (cracker), 0.086700 (sugar box), 0.031850 radius (tomato soup can), 0.090000 (mustard bottle), 0.083200 (gelatin box), 0.095600 (SPAM)
    x_sample = np.arange(-x_upper_range, x_upper_range, 0.15)
    print("x_sample", x_sample)
    # max y collisions of the ycb objects: 0.207400 (cracker), 0.170300 (sugar box), 0.099900 length (tomato soup can), 0.160300 (mustard bottle), 0.067100 (gelatin box), 0.077500 (SPAM)
    y_sample = np.arange(-y_upper_range, y_upper_range, 0.20)
    print("y_sample", y_sample)
    while len(x_points) < num_elements:
        poss_point_x = np.random.choice(x_sample)
        poss_point_y = np.random.choice(y_sample)
        if distance_thres_point(poss_point_x, poss_point_y, x_points, y_points, 0.15):
            x_points.append(poss_point_x)
            y_points.append(poss_point_y)
            print(
                "appended:", "x_points_append:", x_points, "y_points_append:", y_points
            )
    return x_points, y_points

def add_cameras_at_tables():
    camera_poses_W = [
        RigidTransform(
            R=RotationMatrix([
                [-0.008426572055398018, 0.49998224790571183, -0.865994656255191],
                [0.9999644958114239, 0.004213286027699008, -0.007297625466794737],
                [0.0, -0.8660254037844386, -0.49999999999999983],
            ]),
            p=[0.5567144688632728, -0.003735510093671053, 0.495],
        ),
        RigidTransform(
            R=RotationMatrix([
                [0.7564493864543211, -0.32702611735642795, 0.5664258506633156],
                [-0.6540522347128561, -0.3782246932271605, 0.6551043853465944],
                [0.0, -0.8660254037844387, -0.4999999999999999],
            ]),
            p=[-0.4067672805262673, -0.5122634135249003, 0.495],
        ),
        RigidTransform(
            R=RotationMatrix([
                [-0.8214529060279898, -0.28513817842327355, 0.49387381220674975],
                [-0.5702763568465472, 0.4107264530139948, -0.7113990846327904],
                [0.0, -0.8660254037844387, -0.4999999999999999],
            ]),
            p=[-0.35091572089593653, 0.4881919030929625, 0.495],
        )
    ]


def random_object_spawn(
    bodies_list, x_upper_bound, y_upper_bound, plant, plant_context, table, table_height
):
    const = 0
    if table == "left":
        const = -2
    if table == "right":
        const = 2

    z = 0.2 + table_height
    num_elements = len(bodies_list)
    print("number of objects for table", num_elements)
    x_points, y_points = generate_random_with_min_dist(
        x_upper_bound, y_upper_bound, num_elements
    )
    print("x_points", x_points, "y_points", y_points)
    body_index = 0
    for body in bodies_list:
        random_z_theta = random.uniform(0, 2 * np.pi)
        # print(random_z_theta)
        random_z_rotation = RotationMatrix.MakeZRotation(random_z_theta)
        # print("random z rotation:",random_z_rotation)
        tf = RigidTransform(
            # UniformlyRandomRotationMatrix(generator), #completely random rotation
            random_z_rotation,
            # [rng.uniform(-x_upper_bound, x_upper_bound), const + rng.uniform(-y_upper_bound, y_upper_bound), z],
            [x_points[body_index], const + y_points[body_index], z],
        )
        plant.SetFreeBodyPose(plant_context, plant.get_body(body), tf)
        z += 0.01
        body_index += 1

def clutter_gen(
        meshcat,
        rng,
        table_height=0.2,
        add_spot=True,
        debug=False,
        simulation_time=-1,
        add_fixed_cameras=True
):
    scenario_data = f"""
cameras:
    table_top_camera:
        name: camera0
        depth: True
        X_PB:
            base_frame: camera0::base
            translation: [0, 0, {-0.05 + table_height}]
            rotation: !Rpy {{ deg: [-90, 0, 0] }}
    """
    if add_spot:
        # camera welded to the chest of Spot
        scenario_data += """
    spot_camera:
        name: spot_camera
        depth: True
        X_PB:
            translation: [0, 0.05, 0]
            base_frame: spot_camera::base
            rotation: !Rpy { deg: [-90, 0, 0] }
    """

    if add_fixed_cameras:
        # Add cameras at the fixed looking locations around each table
        scenario_data += get_camera_generator_str()

    scenario_data += f"""       
directives:
- add_model:
    name: camera0
    file: package://manipulation/camera_box.sdf
- add_weld:
    parent: world
    child: camera0::base
    X_PC:
        translation: [0, 1.75, 1.0]
        rotation: !Rpy {{ deg: [-75, 0, 0] }}
    
- add_model:
    name: table_top0
    file: package://sponana/table_top.sdf

- add_weld:
    parent: world
    child: table_top0::table_top_center
    X_PC:
        translation: [0, 0, {table_height}]

- add_model:
    name: table_top1
    file: package://sponana/table_top.sdf

- add_weld:
    parent: world
    child: table_top1::table_top_center
    X_PC:
        translation: [0, 2.0, {table_height}]

- add_model:
    name: table_top2
    file: package://sponana/table_top.sdf

- add_weld:
    parent: world
    child: table_top2::table_top_center
    X_PC:
        translation: [0, -2.0, {table_height}]

# Walls
- add_model:
    name: table_top4
    file: package://sponana/table_top4.sdf
- add_weld:
    parent: world
    child: table_top4::table_top4_center

- add_model:
    name: table_top5
    file: package://sponana/table_top5.sdf
- add_weld:
    parent: world
    child: table_top5::table_top5_center

- add_model:
    name: table_top6
    file: package://sponana/table_top6.sdf
- add_weld:
    parent: world
    child: table_top6::table_top6_center

- add_model:
    name: table_top7
    file: package://sponana/table_top7.sdf
- add_weld:
    parent: world
    child: table_top7::table_top7_center

- add_model:
    name: floor
    file: package://sponana/platform.sdf
- add_weld:
    parent: world
    child: floor::platform_center

- add_model:
    name: back_wall
    file: package://sponana/table_top9.sdf
- add_weld:
    parent: world
    child: back_wall::table_top9_center
    """

    for i in range(22 if running_as_notebook else 2):
        object_num = rng.integers(0, len(ycb))
        scenario_data += f"""
- add_model:
    name: thing{i}
    file: package://manipulation/hydro/{ycb[object_num]} 
"""
    scenario_data += f"""
- add_model:
    name: banana
    file: package://sponana/banana/banana.sdf
    default_free_body_pose:
        banana: 
            translation: [0, 0, 1]
            rotation: !Rpy {{ deg: [0, 0, 0] }}    
"""

    if add_spot:
        scenario_data += """
- add_model:
    name: spot
    file: package://manipulation/spot/spot_with_arm_and_floating_base_actuators.urdf
    default_joint_positions:
        # fold the arm
        arm_sh1: [-3.1]
        arm_el0: [3.1]
        # initial position
        base_x: [1.0]
        base_rz: [3.14]

- add_model:
    name: spot_camera
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: spot::body
    child: spot_camera::base
    X_PC:
        translation: [0.4, 0, 0]
        # pointed slightly downward (by 30 degrees)
        rotation: !Rpy { deg: [-30, 0, -90] }

model_drivers:
    spot: !InverseDynamicsDriver {}
"""

    builder = DiagramBuilder()
    scenario = load_scenario(data=scenario_data)
    station = builder.AddSystem(
        MakeHardwareStation(
            scenario, meshcat, parser_preload_callback=sponana.utils.configure_parser
        )
    )

    if add_spot:
        spot_plant = station.GetSubsystemByName(
            "spot.controller"
        ).get_multibody_plant_for_control()
        spot_controller = builder.AddSystem(SpotController(spot_plant, meshcat=meshcat))
        builder.Connect(
            spot_controller.get_output_port(),
            station.GetInputPort("spot.desired_state"),
        )

        # Get camera and table poses
        spot_camera_config = scenario.cameras["spot_camera"]
        camera_pose_extractor = add_camera_pose_extractor(
            spot_camera_config, station, builder
        )
        table_pose_extractors = [
            add_body_pose_extractor(f"table_top{i}", "table_top_link", station, builder)
            for i in range(3)
        ]

        # Perception system (Banan Spotter) (placeholder for now)
        spot_camera = station.GetSubsystemByName("rgbd_sensor_spot_camera")
        banana_spotter = builder.AddNamedSystem(
            "banana_spotter",
            BananaSpotter(spot_camera, num_tables=len(table_pose_extractors)),
        )
        builder.Connect(
            station.GetOutputPort("spot_camera.rgb_image"),
            banana_spotter.get_color_image_input_port(),
        )
        builder.Connect(
            station.GetOutputPort("spot_camera.depth_image"),
            banana_spotter.get_depth_image_input_port(),
        )
        builder.Connect(
            camera_pose_extractor.get_output_port(),
            banana_spotter.get_camera_pose_input_port(),
        )
        for i, pose_extractor in enumerate(table_pose_extractors):
            builder.Connect(
                pose_extractor.get_output_port(),
                banana_spotter.get_table_pose_input_port(i),
            )

        if debug:
            # Connect debugger
            spot_camera = station.GetSubsystemByName("rgbd_sensor_spot_camera")
            debugger = builder.AddNamedSystem(
                "debug_logger",
                DebugLogger(
                    spot_camera, meshcat, num_tables=len(table_pose_extractors)
                ),
            )
            builder.Connect(
                station.GetOutputPort("spot_camera.rgb_image"),
                debugger.get_color_image_input_port(),
            )
            builder.Connect(
                station.GetOutputPort("spot_camera.depth_image"),
                debugger.get_depth_image_input_port(),
            )
            builder.Connect(
                camera_pose_extractor.get_output_port(),
                debugger.get_camera_pose_input_port(),
            )
            for i, pose_extractor in enumerate(table_pose_extractors):
                builder.Connect(
                    pose_extractor.get_output_port(),
                    debugger.get_table_pose_input_port(i),
                )

    diagram = builder.Build()

    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant = station.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)

    # print("plant Floating Base", set(list(plant.GetFloatingBaseBodies())[0:3]))
    random_z_theta = random.uniform(0, 2 * np.pi)
    print(random_z_theta)
    random_z_rotation = RotationMatrix.MakeZRotation(random_z_theta)
    print("random z rotation:", random_z_rotation)
    # print(type(plant.GetFloatingBaseBodies()))
    floating_base_bodies_list = list(plant.GetFloatingBaseBodies())
    len_bodies = len(floating_base_bodies_list)

    center_table_bodies = set(floating_base_bodies_list[0 : len_bodies // 3])
    x_upper_bound = 0.20
    y_upper_bound = 0.30
    random_object_spawn(
        center_table_bodies, x_upper_bound, y_upper_bound, plant, plant_context, "", table_height
    )

    right_table_bodies = set(
        floating_base_bodies_list[len_bodies // 3 : len_bodies // 3 * 2]
    )
    random_object_spawn(
        right_table_bodies, x_upper_bound, y_upper_bound, plant, plant_context, "right", table_height
    )

    left_table_bodies = set(floating_base_bodies_list[len_bodies // 3 * 2 :])
    random_object_spawn(
        left_table_bodies, x_upper_bound, y_upper_bound, plant, plant_context, "left", table_height
    )

    sponana.utils.run_simulation(simulator, meshcat, finish_time=simulation_time)
    return simulator, diagram