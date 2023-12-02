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

###Added in sim.py for FSM
##This is just notes file for reference on double monitors


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