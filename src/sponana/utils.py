from pathlib import Path
from typing import Optional

import numpy as np
import pydot
from IPython.display import SVG, display
from manipulation import ConfigureParser
from pydrake.all import Context, Diagram, Meshcat, PackageMap, Parser, Simulator


def configure_parser(parser: Parser):
    """A helper function that registers `manipulation` package, Spot model,
    as well as additional models from SPONANA's `models` directory to the
    given parser"""
    # Add the manipulation/package.xml index to the given Parser
    ConfigureParser(parser)
    # Additional Spot metadata
    parser.package_map().AddRemote(
        package_name="spot_description",
        params=PackageMap.RemoteParams(
            urls=[
                f"https://github.com/bdaiinstitute/spot_ros2/archive/d429947a1df842ec38f8c6099dde9501945090d6.tar.gz"
            ],
            sha256=("e4dd471be4e7e822a12afcfd6a94ce7ecbb39e2d4ea406779a96e146a607bf53"),
            strip_prefix="spot_ros2-d429947a1df842ec38f8c6099dde9501945090d6/spot_description/",
        ),
    )
    # Add Sponana to the parser
    models_path = Path(__file__).parent / "models"
    parser.package_map().Add("sponana", str(models_path.resolve()))


def visualize_diagram(diagram: Diagram, max_depth: Optional[int] = None):
    display(
        SVG(
            pydot.graph_from_dot_data(diagram.GetGraphvizString(max_depth=max_depth))[
                0
            ].create_svg()
        )
    )


def run_simulation(simulator: Simulator, meshcat: Meshcat, finish_time=2.0):
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)

    if finish_time <= 0:
        # keep simulation running
        meshcat.AddButton("Stop Simulation", "Escape")
        print("Press Escape to stop the simulation")
        while meshcat.GetButtonClicks("Stop Simulation") < 1:
            simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)
        meshcat.DeleteButton("Stop Simulation")
    else:
        # run similator for a fixed duration and publish recording
        meshcat.StartRecording()
        simulator.AdvanceTo(finish_time)
        meshcat.PublishRecording()


def set_spot_positions(
    spot_state: np.ndarray,
    diagram: Diagram,
    root_context: Context,
    visualize: bool = True,
):
    """Update the Spot's joint positions to the given `spot_state` (a 10-dim vector) in the
    diagram. If `visualize` is True, the updated diagram will be visualized in Meshcat.
    """
    plant = diagram.GetSubsystemByName("station").GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(root_context)
    plant.SetPositions(plant_context, plant.GetModelInstanceByName("spot"), spot_state)

    if visualize:
        diagram_context = diagram.GetMyContextFromRoot(root_context)
        diagram.ForcedPublish(diagram_context)
