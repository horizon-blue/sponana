"""
A collection of different table specs for the final report
"""
import dataclasses

import numpy as np

from .sim import TableSceneSpec

_empty_table = TableSceneSpec(
    has_banana=False,
    banana_contact_params=(0, 0, 0, 0),  # doesn't matter
    n_objects=0,
    object_type_indices=[],
    object_contact_params=[],
)
_single_banana = TableSceneSpec(
    has_banana=True,
    banana_contact_params=(0, 0, 0, 0),
    n_objects=0,
    object_type_indices=[4, 1],  # gelatin, cracker box
    object_contact_params=[(0.02, -0.08, 0, 2), (-0.1, 0.1, 0, 2)],
)
_occlusions = TableSceneSpec(
    has_banana=False,
    banana_contact_params=(-0.12, 0.0, 3 * np.pi / 2, 0),
    n_objects=3,
    # Cracker boxes
    object_type_indices=[0, 0, 0],
    object_contact_params=[
        (0.02, -0.08, 0, 2),
        (0.1, 0.1, 0, 2),
        (-0.12, -0.2, np.pi / 4, 2),
    ],
)

two_empty_rooms_with_a_banana = [_empty_table, _empty_table, _single_banana]
occluded_by_a_cracker_box = [
    _occlusions,
    _occlusions,
    dataclasses.replace(_occlusions, has_banana=True),
]
