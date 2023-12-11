import warnings

from .dummy_banana_spotter import DummyBananaSpotter
from .utils import add_body_pose_extractor, add_camera_pose_extractor

try:
    from .banana_spotter_bayes3d import BananaSpotterBayes3D
except ImportError:
    warnings.warn(
        "Failed to import banana spotter. Have you installed bayes3d? "
        "Falling back to use the dummy banana spotter..."
    )
    BananaSpotterBayes3D = DummyBananaSpotter
