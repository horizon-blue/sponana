from pydrake.all import Parser, PackageMap
from manipulation import ConfigureParser
from pathlib import Path

MODELS_PATH = Path(__file__).parent / "models"
PACKAGE_XMLS = [
    str(path.resolve())
    for path in [
        MODELS_PATH / "package.xml",  # SPONANA models
        MODELS_PATH / "spot_ros2/spot_description/package.xml",
    ]
]


def configure_parser(parser: Parser):
    """A helper function that registers `manipulation` package, Spot model,
    as well as additional models from SPONANA's `models` directory to the
    given parser"""
    # Add the manipulation/package.xml index to the given Parser
    ConfigureParser(parser)
    for xml in PACKAGE_XMLS:
        parser.package_map().AddPackageXml(xml)
