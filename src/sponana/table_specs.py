"""
A collection of different table specs for the final report
"""
import dataclasses

import numpy as np


@dataclasses.dataclass
class TableSceneSpec:
    """
    Description of the contents of one table top.  `None` values will be filled in via random generation
    in `create_and_run_simulation`.

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
    object_contact_params=[(-0.12, -0.13, 0.05, 2), (-0.1, 0.1, 0, 2)],
)
_half_occlusions = TableSceneSpec(
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
_full_occlusions = TableSceneSpec(
    has_banana=False,
    banana_contact_params=(-0.12, 0.0, 3 * np.pi / 2, 0),
    n_objects=5,
    # Cracker boxes
    object_type_indices=[0, 0, 0, 0, 0],
    object_contact_params=[
        (0.02, -0.08, 0, 2),
        (0.1, 0.1, 0, 2),
        (-0.12, -0.2, np.pi / 4, 2),
        (-0.15, 0.08, -np.pi / 6, 2),
        (-0.08, 0.23, -np.pi / 2, 2),
    ],
)

two_empty_rooms_with_a_banana = [_empty_table, _empty_table, _single_banana]
half_occluded_by_a_cracker_box = [
    _half_occlusions,
    _half_occlusions,
    dataclasses.replace(_half_occlusions, has_banana=True),
]
full_occluded_by_a_cracker_box = [
    _full_occlusions,
    _full_occlusions,
    dataclasses.replace(_full_occlusions, has_banana=True),
]
default_table_specs = [
    TableSceneSpec(has_banana=False),
    TableSceneSpec(has_banana=False),
    TableSceneSpec(has_banana=True),
]
mix_rooms = [
    dataclasses.replace(_single_banana, n_objects=1),
    TableSceneSpec(has_banana=False),
    _half_occlusions,
]
