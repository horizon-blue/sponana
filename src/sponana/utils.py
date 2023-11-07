from pydrake.all import Parser, PackageMap
from manipulation import ConfigureParser
from pathlib import Path


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
    models_path = Path(__file__).parent
    parser.package_map().Add("sponana", str(models_path.resolve()))
