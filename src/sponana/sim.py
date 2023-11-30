import copy
import math
import random
from dataclasses import dataclass

import numpy as np
from IPython.display import HTML, display
from manipulation import ConfigureParser, running_as_notebook
from manipulation.scenarios import AddShape, ycb
from manipulation.station import MakeHardwareStation, load_scenario
from matplotlib import pyplot as plt
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    ConnectPlanarSceneGraphVisualizer,
    DiagramBuilder,
    FixedOffsetFrame,
    JointIndex,
    LogVectorOutput,
    Parser,
    PlanarJoint,
    RandomGenerator,
    RigidTransform,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    UniformlyRandomRotationMatrix,
)
from pydrake.common import temp_directory
from pydrake.geometry import StartMeshcat
from pydrake.systems.analysis import Simulator
from pydrake.visualization import ModelVisualizer

# sponana/src/sponana/utils.py
import sponana.utils
from sponana.controller import SpotController
from sponana.debug_logger import DebugLogger
from sponana.hardcoded_cameras import get_camera_generator_str
from sponana.perception import (
    BananaSpotter,
    add_body_pose_extractor,
    add_camera_pose_extractor,
)
from sponana.planner import Navigator
from sponana.fsm import finite_state_machine
add_finite_state_machine = False

@dataclass
class TableSceneSpec:
    """
    Description of the contents of one table top.  `None` values will be filled in via random generation
    in `clutter_gen`.

    - n_objects is the number of non-banana objects.
    - object_type_indices is a list of indices into `manipulation.scenarios.ycb`.
    - object_contact_params is a list of tuples (x, y, theta, face). face is in (0, 1, 2)
        and indicates which face of the object is in contact with the table.
    """

    has_banana: bool = False
    banana_contact_params: tuple = None
    n_objects: int = None
    object_type_indices: list = None
    object_contact_params: list = None


def clutter_gen(
    meshcat,
    rng,
    add_spot=True,
    debug=False,
    simulation_time=-1,
    add_fixed_cameras=True,
    enable_arm_ik=True,
    table_specs=[
        TableSceneSpec(has_banana=False),
        TableSceneSpec(has_banana=True),
        TableSceneSpec(has_banana=False),
    ],
    use_teleop=True,
):
    """
    Generate a Sponana environment consistent with the provided `table_specs`.
    """

    # Randomly generate specifications for the tables if any are incomplete
    table_specs = concretize_table_specs(table_specs, rng)
    print(table_specs)

    scenario_data = f"""
cameras:
    table_top_camera:
        name: camera0
        depth: True
        X_PB:
            base_frame: camera0::base
            translation: [0, 0, 0.15]
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

    # Visible camera overlooking table 2
    scenario_data += """       
directives:
- add_directives:
    file: package://sponana/scenes/three_rooms_with_tables.dmd.yaml

- add_model:
    name: camera0
    file: package://manipulation/camera_box.sdf
- add_weld:
    parent: world
    child: camera0::base
    X_PC:
        translation: [0, 1.75, 1.0]
        rotation: !Rpy { deg: [-75, 0, 0] }
"""

    # Add distractor objects
    for i, spec in enumerate(table_specs):
        for j in range(spec.n_objects):
            scenario_data += f"""
- add_model:
    name: object{j}_table{i}
    file: package://manipulation/hydro/{ycb[spec.object_type_indices[j]]}
"""

    # Add banana
    scenario_data += f"""
- add_model:
    name: banana
    file: package://sponana/banana/banana.sdf
    default_free_body_pose:
        banana: 
            translation: [0, 0, 1]
            rotation: !Rpy {{ deg: [0, 0, 0] }}    
"""

    # Spot
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
        spot_controller = builder.AddSystem(
            SpotController(
                spot_plant,
                meshcat=meshcat,
                enable_arm_ik=enable_arm_ik,
                use_teleop=use_teleop,
            )
        )
        builder.Connect(
            spot_controller.get_output_port(),
            station.GetInputPort("spot.desired_state"),
        )

        if not use_teleop:
            # planner
            planner = builder.AddNamedSystem("navigator", Navigator(meshcat=meshcat))
            builder.Connect(
                station.GetOutputPort("spot.state_estimated"),
                planner.get_spot_state_input_port(),
            )
            builder.Connect(
                planner.get_output_port(),
                spot_controller.GetInputPort("desired_base_position"),
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

        # Banana pose (using cheat port -- placeholder for now)
        banana_pose_extractor = add_body_pose_extractor(
            "banana", "banana", station, builder
        )
        builder.Connect(
            banana_pose_extractor.get_output_port(),
            spot_controller.GetInputPort("desired_gripper_pose"),
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
            builder.Connect(
                station.GetOutputPort("spot.state_estimated"),
                debugger.get_spot_state_input_port(),
            )
            builder.Connect(
                banana_pose_extractor.get_output_port(),
                debugger.get_banana_pose_input_port(),
            )
        ###Not Done will add
        if add_finite_state_machine:
            fsm = builder.AddNamedSystem("fsm", finite_state_machine(meshcat=meshcat))
            builder.Connect(station.GetOutputPort())

    diagram = builder.Build()

    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant = station.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)

    # Set the poses of the YCB objects
    for i, spec in enumerate(table_specs):
        object_poses = list(
            map((lambda cps: cps_to_pose(cps, i)), spec.object_contact_params)
        )
        for j in range(spec.n_objects):
            plant.SetFreeBodyPose(
                plant_context,
                plant.get_body(
                    plant.GetBodyIndices(
                        plant.GetModelInstanceByName(f"object{j}_table{i}")
                    )[0]
                ),
                object_poses[j],
            )

    # Set the pose of the banana
    banana_table_idx = next(
        (i for i, spec in enumerate(table_specs) if spec.has_banana), None
    )
    print("banana_table_idx", banana_table_idx)
    print("spec[banana_table_idx].has_banana", table_specs[banana_table_idx].has_banana)
    banana_pose = cps_to_pose(
        table_specs[banana_table_idx].banana_contact_params, banana_table_idx
    )
    plant.SetFreeBodyPose(
        plant_context,
        plant.get_body(
            plant.GetBodyIndices(plant.GetModelInstanceByName(f"banana"))[0]
        ),
        banana_pose,
    )

    # Run the simulation
    sponana.utils.run_simulation(simulator, meshcat, finish_time=simulation_time)
    return simulator, diagram


### Utils for object generation ###
def concretize_table_specs(table_specs, rng):
    """
    Fill in `None` values in a list of `TableSceneSpec`s via random generation.
    Also ensure that exactly one table has a banana.
    """
    if not any(spec.has_banana for spec in table_specs):
        idx = rng.choice(len(table_specs))
    else:
        idx = next((i for i, spec in enumerate(table_specs) if spec.has_banana), None)

    complete_specs = []
    for i, spec in enumerate(table_specs):
        spec = copy.deepcopy(spec)
        spec.has_banana = i == idx
        if spec.has_banana and spec.banana_contact_params is None:
            spec.banana_contact_params = generate_contact_params(rng, 1)[0]
        if spec.n_objects is None:
            spec.n_objects = rng.integers(1, 5)
        if spec.object_type_indices is None:
            spec.object_type_indices = rng.choice(len(ycb), spec.n_objects)
        if spec.object_contact_params is None:
            spec.object_contact_params = generate_contact_params(rng, spec.n_objects)
        complete_specs.append(spec)

    return complete_specs


# Randomly generate contact parameters for n_objects objects.
# Returns a list of tuples (x, y, theta, face) where face is the face of the object that is in contact with the table.
# Face will be 0, 1, or 2.
def generate_contact_params(
    rng,
    n_objects,
    x_upper_bound=0.20,
    y_upper_bound=0.30,
):
    x_points, y_points = generate_random_with_min_dist(
        rng, x_upper_bound - 0.1, y_upper_bound - 0.1, n_objects
    )
    thetas = [rng.uniform(0, 2 * np.pi) for _ in range(n_objects)]
    faces = [rng.choice(3) for _ in range(n_objects)]
    return list(zip(x_points, y_points, thetas, faces))


# Contact parameters = (x, y, theta, face)
# Convert this to a pose (RigidTransform).
def cps_to_pose(cps, table_idx):
    x = cps[0]
    y = cps[1]
    theta = cps[2]
    face = cps[3]

    if face == 0:
        extra_rot = RotationMatrix.Identity()
    elif face == 1:
        extra_rot = RotationMatrix.MakeXRotation(np.pi / 2)
    elif face == 2:
        extra_rot = RotationMatrix.MakeYRotation(np.pi / 2)

    if table_idx == 0:
        y = y + 2.0
    elif table_idx == 2:
        y = y - 2.0

    return RigidTransform(RotationMatrix.MakeZRotation(theta) @ extra_rot, [x, y, 0.4])


def generate_random_with_min_dist(rng, x_upper_range, y_upper_range, num_elements):
    x_points = []
    y_points = []
    # max x collisions of the ycb objects: 0.158000 (cracker), 0.086700 (sugar box), 0.031850 radius (tomato soup can), 0.090000 (mustard bottle), 0.083200 (gelatin box), 0.095600 (SPAM)
    x_sample = np.arange(-x_upper_range, x_upper_range, 0.15)
    print("x_sample", x_sample)
    # max y collisions of the ycb objects: 0.207400 (cracker), 0.170300 (sugar box), 0.099900 length (tomato soup can), 0.160300 (mustard bottle), 0.067100 (gelatin box), 0.077500 (SPAM)
    y_sample = np.arange(-y_upper_range, y_upper_range, 0.20)
    print("y_sample", y_sample)
    while len(x_points) < num_elements:
        poss_point_x = rng.choice(x_sample)
        poss_point_y = rng.choice(y_sample)
        if distance_thres_point(poss_point_x, poss_point_y, x_points, y_points, 0.15):
            x_points.append(poss_point_x)
            y_points.append(poss_point_y)
            print(
                "appended:", "x_points_append:", x_points, "y_points_append:", y_points
            )
    return x_points, y_points


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
