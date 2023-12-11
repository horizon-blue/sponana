# This file contains strings for constructing different enviornments used in 
# mental simulations the robot uses for grasping.

DEFAULT_TARGET_OBJ_PATH = "package://sponana/banana/banana.sdf"
DEFAULT_TARGET_OBJ_LINK = "banana"
DEFAULT_TARGET_OBJ_RPY_STR = "[0, 0, 0]"

SUGAR_BOX_TARGET_PATH = "package://sponana/grasping/004_sugar_box.sdf"
SUGAR_BOX_TARGET_LINK = "base_link_sugar"
SUGAR_BOX_RPY_STR = "[90, 90, 90]"

def spot_and_target_str(
        target_obj_path=DEFAULT_TARGET_OBJ_PATH,
        target_obj_link=DEFAULT_TARGET_OBJ_LINK,
        target_obj_rpy_str=DEFAULT_TARGET_OBJ_RPY_STR
    ):
    """
    String describing an enviornment with the spot robot and the target object.
    - target_obj_path is the path to an SDF or URDF file for the target object (e.g. banana).
    - target_obj_link is the base link name of the target object.
    - target_obj_rpy_str is a string describing the target object's roll pitch yaw orientation in degrees.
    """
    return """
directives:
- add_frame:
    name: banana_origin
    X_PF:
        base_frame: world
        rotation: !Rpy { deg: """ + target_obj_rpy_str + " }" + f"""
        translation: [0, 0, 0.515]

- add_model:
    name: banana
    file: {target_obj_path}

- add_weld:
    parent: banana_origin
    child: banana::{target_obj_link}

- add_model:
    name: spot
    file: package://sponana/spot.urdf
    default_joint_positions:
        # fold the arm
        arm_sh1: [-3.1]
        arm_el0: [3.1]
        # initial position
        base_x: [1.0]
        base_rz: [3.14]

- add_model:
    name: table_top0
    file: package://sponana/table_top.sdf

- add_weld:
    parent: world
    child: table_top0::table_top_center
    X_PC:
        translation: [0, 0, 0.47]
"""

def gripper_only_str():
    return f"""
directives:
- add_model:
    name: _gripper
    file: package://sponana/grasping/spot_gripper.urdf
"""

def gripper_and_target_str(
        target_obj_path=DEFAULT_TARGET_OBJ_PATH,
        target_obj_link=DEFAULT_TARGET_OBJ_LINK,
        target_obj_rpy_str=DEFAULT_TARGET_OBJ_RPY_STR
    ):
    """
    String describing an enviornment with the spot gripper and the target object.
    - target_obj_path is the path to an SDF or URDF file for the target object (e.g. banana).
    - target_obj_link is the base link name of the target object.
    - target_obj_rpy_str is a string describing the target object's roll pitch yaw orientation in degrees.
    """
    return f"""
directives:
{_target_and_cameras_str(target_obj_path=target_obj_path, target_obj_link=target_obj_link, target_obj_rpy_str=target_obj_rpy_str)}

- add_model:
    name: _gripper
    file: package://sponana/grasping/spot_gripper.urdf

- add_model:
    name: table_top0
    file: package://sponana/table_top.sdf

- add_weld:
    parent: world
    child: table_top0::table_top_center
    X_PC:
        translation: [0, 0, -0.01]
"""

def target_and_cameras_str(
        target_obj_path=DEFAULT_TARGET_OBJ_PATH,
        target_obj_link=DEFAULT_TARGET_OBJ_LINK,
        target_obj_rpy_str=DEFAULT_TARGET_OBJ_RPY_STR
    ):
    """
    String describing an enviornment with the target object and three cameras.
    """
    return f"""
directives:
{_target_and_cameras_str(target_obj_path=target_obj_path, target_obj_link=target_obj_link, target_obj_rpy_str=target_obj_rpy_str)}
"""

def _target_and_cameras_str(
        target_obj_path=DEFAULT_TARGET_OBJ_PATH, 
        target_obj_link=DEFAULT_TARGET_OBJ_LINK,
        target_obj_rpy_str=DEFAULT_TARGET_OBJ_RPY_STR
    ):
    return """
- add_frame:
    name: banana_origin
    X_PF:
        base_frame: world
        rotation: !Rpy { deg: """ + target_obj_rpy_str + """}
        translation: [0, 0, 0.045]

- add_model:
    name: banana
    file: """ + target_obj_path + """

- add_weld:
    parent: banana_origin
    child: banana::""" + target_obj_link + """

- add_frame:
    name: camera0_staging
    X_PF:
        base_frame: world
        rotation: !Rpy { deg: [0, 0, 15.0]}

- add_frame:
    name: camera1_staging
    X_PF:
        base_frame: world
        rotation: !Rpy { deg: [0, 0, 130.0]}

- add_frame:
    name: camera2_staging
    X_PF:
        base_frame: world
        rotation: !Rpy { deg: [0, 0, 245.0]}

- add_frame:
    name: camera0_origin
    X_PF:
        base_frame: camera0_staging
        rotation: !Rpy { deg: [-100.0, 0, 90.0]}
        translation: [.5, 0, .2]

- add_model:
    name: camera0
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: camera0_origin
    child: camera0::base

- add_frame:
    name: camera1_origin
    X_PF:
        base_frame: camera1_staging
        rotation: !Rpy { deg: [-100.0, 0, 90.0]}
        translation: [.5, 0, .2]

- add_model:
    name: camera1
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: camera1_origin
    child: camera1::base

- add_frame:
    name: camera2_origin
    X_PF:
        base_frame: camera2_staging
        rotation: !Rpy { deg: [-100.0, 0, 90.0]}
        translation: [.5, 0, .2]

- add_model:
    name: camera2
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: camera2_origin
    child: camera2::base
"""