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