import copy
import logging
import math
from dataclasses import dataclass

import numpy as np
from manipulation.scenarios import ycb
from manipulation.station import MakeHardwareStation, load_scenario
from pydrake.all import (
    ConstantVectorSource,
    DiagramBuilder,
    RigidTransform,
    RotationMatrix,
    Simulator,
)

import sponana.utils
from sponana.controller import SpotController
from sponana.debug_logger import DebugLogger
from sponana.fsm import FiniteStateMachine
from sponana.grasping import Grasper
from sponana.hardcoded_cameras import get_camera_generator_str
from sponana.perception import (
    BananaSpotter,
    add_body_pose_extractor,
    add_camera_pose_extractor,
)
from sponana.planner import Navigator

from .hardcoded_cameras import get_base_positions_for_hardcoded_cameras

logger = logging.getLogger(__name__)


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
    add_debug_logger=False,
    simulation_time=-1,
    add_fixed_cameras=True,
    table_specs=[
        TableSceneSpec(has_banana=False),
        TableSceneSpec(has_banana=False),
        TableSceneSpec(has_banana=True),
    ],
    use_teleop=True,
    starting_position=[3.0, 7.0, -1.57],
):
    """
    Generate a Sponana environment consistent with the provided `table_specs`.
    """

    # Randomly generate specifications for the tables if any are incomplete
    table_specs = concretize_table_specs(table_specs, rng)
    logger.debug(table_specs)

    scenario_data = """
cameras:
    # camera welded to the chest of Spot
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
- add_directives:
    file: package://sponana/scenes/three_rooms_with_tables.dmd.yaml

- add_model:
    name: spot
    file: package://sponana/spot.urdf
    default_joint_positions:
        # fold the arm
        arm_sh1: [-3.1]
        arm_el0: [3.1]
        # initial position
        base_x: [{starting_position[0]}]
        base_y: [{starting_position[1]}]
        base_rz: [{starting_position[2]}]

- add_model:
    name: spot_camera
    file: package://manipulation/camera_box.sdf
"""
    # Add distractor objects
    for i, spec in enumerate(table_specs):
        for j in range(spec.n_objects):
            scenario_data += f"""
- add_model:
    name: object{j}_table{i}
    file: package://manipulation/hydro/{ycb[spec.object_type_indices[j]]}
"""
    scenario_data += """
- add_model:
    name: banana
    file: package://sponana/banana/banana.sdf

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
        sponana.utils.MakeSponanaHardwareStation(scenario, meshcat)
    )

    spot_plant = station.GetSubsystemByName(
        "spot.controller"
    ).get_multibody_plant_for_control()
    spot_controller = builder.AddSystem(
        SpotController(
            spot_plant,
            meshcat=meshcat,
            use_teleop=use_teleop,
        )
    )
    builder.Connect(
        spot_controller.get_output_port(),
        station.GetInputPort("spot.desired_state"),
    )

    ##########################################################
    # Create and Add Systems to the Diagram
    ##########################################################
    table_pose_extractors = [
        add_body_pose_extractor(f"table_top{i}", "table_top_link", station, builder)
        for i in range(3)
    ]
    spot_camera = station.GetSubsystemByName("rgbd_sensor_spot_camera")
    spot_camera_config = scenario.cameras["spot_camera"]
    camera_pose_extractor = add_camera_pose_extractor(
        spot_camera_config, station, builder
    )
    banana_spotter = builder.AddNamedSystem(
        "banana_spotter",
        BananaSpotter(spot_camera, num_tables=len(table_pose_extractors)),
    )
    # Banana pose (using cheat port -- placeholder for now)
    banana_pose_extractor = add_body_pose_extractor(
        "banana", "banana", station, builder
    )

    if not use_teleop:
        # Additional systems for autonomous mode
        navigator = builder.AddNamedSystem("navigator", Navigator(meshcat=meshcat))
        grasper = builder.AddNamedSystem("grasper", Grasper())
        fsm = builder.AddNamedSystem(
            "finite_state_machine",
            FiniteStateMachine(
                target_base_positions=get_base_positions_for_hardcoded_cameras()
            ),
        )

    ##########################################################
    # Connect the I/O Ports of the Systems
    ##########################################################
    builder.Connect(
        station.GetOutputPort("spot_camera.rgb_image"),
        banana_spotter.get_color_image_input_port(),
    )
    builder.Connect(
        station.GetOutputPort("spot_camera.depth_image"),
        banana_spotter.get_depth_image_input_port(),
    )
    for i, pose_extractor in enumerate(table_pose_extractors):
        builder.Connect(
            pose_extractor.get_output_port(),
            banana_spotter.get_table_pose_input_port(i),
        )
    if not use_teleop:
        # Navigator
        builder.Connect(
            station.GetOutputPort("spot.state_estimated"),
            navigator.get_spot_state_input_port(),
        )
        builder.Connect(
            navigator.get_base_position_output_port(),
            spot_controller.GetInputPort("desired_base_position"),
        )
        builder.Connect(
            fsm.get_do_rrt_output_port(),
            navigator.get_do_rrt_input_port(),
        )
        builder.Connect(
            navigator.get_done_rrt_output_port(), fsm.get_camera_reached_input_port()
        )
        builder.Connect(
            fsm.get_target_base_position_output_port(),
            navigator.get_target_position_input_port(),
        )

        # Perception (Banana Spotter)
        builder.Connect(
            fsm.get_check_banana_output_port(),
            banana_spotter.get_check_banana_input_port(),
        )
        builder.Connect(
            camera_pose_extractor.get_output_port(),
            banana_spotter.get_camera_pose_input_port(),
        )
        builder.Connect(
            banana_spotter.get_found_banana_output_port(),
            fsm.get_see_banana_input_port(),
        )

        # Grasper (banana pose -> gripper joint positions)
        builder.Connect(
            banana_pose_extractor.get_output_port(),
            grasper.get_banana_pose_input_port(),
        )
        # not too sure what's the reset time for... let's fix something for now
        time_zero_source = builder.AddNamedSystem(
            "time_zero_source", ConstantVectorSource(np.array([1.0]))
        )
        builder.Connect(
            time_zero_source.get_output_port(), grasper.get_reset_time_input_port()
        )
        builder.Connect(
            grasper.get_banana_grasped_output_port(), fsm.get_has_banana_input_port()
        )
        builder.Connect(
            station.GetOutputPort("spot.state_estimated"),
            grasper.get_spot_state_input_port(),
        )
        builder.Connect(
            fsm.get_grasp_banana_output_port(), grasper.get_do_grasp_input_port()
        )
        builder.Connect(
            grasper.get_arm_position_output_port(),
            spot_controller.GetInputPort("desired_arm_position"),
        )

    if add_debug_logger:
        # Connect debugger
        spot_camera = station.GetSubsystemByName("rgbd_sensor_spot_camera")
        debugger = builder.AddNamedSystem(
            "debug_logger",
            DebugLogger(spot_camera, meshcat, num_tables=len(table_pose_extractors)),
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
    logger.debug(f"banana_table_idx: {banana_table_idx}")
    logger.debug(
        f"spec[banana_table_idx].has_banana: {table_specs[banana_table_idx].has_banana}"
    )
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
        y = y + 4.0
    elif table_idx == 2:
        y = y - 4.0

    return RigidTransform(RotationMatrix.MakeZRotation(theta) @ extra_rot, [x, y, 0.4])


def generate_random_with_min_dist(rng, x_upper_range, y_upper_range, num_elements):
    x_points = []
    y_points = []
    # max x collisions of the ycb objects: 0.158000 (cracker), 0.086700 (sugar box), 0.031850 radius (tomato soup can), 0.090000 (mustard bottle), 0.083200 (gelatin box), 0.095600 (SPAM)
    x_sample = np.arange(-x_upper_range, x_upper_range, 0.15)
    logger.debug(f"x_sample: {x_sample}")
    # max y collisions of the ycb objects: 0.207400 (cracker), 0.170300 (sugar box), 0.099900 length (tomato soup can), 0.160300 (mustard bottle), 0.067100 (gelatin box), 0.077500 (SPAM)
    y_sample = np.arange(-y_upper_range, y_upper_range, 0.20)
    logger.debug(f"y_sample: {y_sample}")
    while len(x_points) < num_elements:
        poss_point_x = rng.choice(x_sample)
        poss_point_y = rng.choice(y_sample)
        if distance_thres_point(poss_point_x, poss_point_y, x_points, y_points, 0.15):
            x_points.append(poss_point_x)
            y_points.append(poss_point_y)
            logger.debug(
                f"appended: x_points_append: {x_points}, y_points_append: {y_points}"
            )
    return x_points, y_points


def distance_thres_point(poss_point_x, poss_point_y, x_points, y_points, r_threshold):
    # logger.debug(f"poss_point_x: {poss_point_x}, poss_point_y: {poss_point_y}")
    for ind in range(len(x_points)):
        dist = math.hypot(
            abs(poss_point_x - x_points[ind]), abs(poss_point_y - y_points[ind])
        )
        logger.debug(f"dist: {dist}")
        if dist < r_threshold:
            return False
    return True
