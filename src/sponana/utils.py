from pathlib import Path
from typing import Optional

import numpy as np
import pydot
from IPython.display import SVG, display
from manipulation import ConfigureParser
from manipulation.station import MakeHardwareStation
from pydrake.all import Context, Diagram, Meshcat, PackageMap, Parser, Simulator

sponana_dir = Path(__file__).parent.absolute()


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
                f"https://github.com/wrangel-bdai/spot_ros2/archive/20965ef7bba98598ee10878c7b54e6ef28a300c6.tar.gz"
            ],
            sha256=("20a4f12896b04cc73e186cf876bf2c7e905ee88f8add8ea51bf52dfc888674b4"),
            strip_prefix="spot_ros2-20965ef7bba98598ee10878c7b54e6ef28a300c6/spot_description/",
        ),
    )
    # Add Sponana to the parser
    models_path = Path(__file__).parent / "models"
    parser.package_map().Add("sponana", str(models_path.resolve()))


def MakeSponanaHardwareStation(scenario, meshcat):
    return MakeHardwareStation(
        scenario,
        meshcat,
        parser_preload_callback=configure_parser,
        package_xmls=[str(sponana_dir / "models/package.xml")],
    )


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
    station: Diagram,
    root_context: Context,
    visualize: bool = True,
):
    """Update the Spot's joint positions to the given `spot_state` (a 10-dim vector) in the
    diagram. If `visualize` is True, the updated diagram will be visualized in Meshcat.
    """
    plant = station.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(root_context)
    plant.SetPositions(plant_context, plant.GetModelInstanceByName("spot"), spot_state)

    if visualize:
        station_context = station.GetMyContextFromRoot(root_context)
        station.ForcedPublish(station_context)
