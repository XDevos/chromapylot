from dataclasses import dataclass, field
from typing import Dict, List, Union

from dataclasses_json import CatchAll, Undefined, dataclass_json
from chromapylot.parameters.utils import set_default


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class MatrixParams:
    """Class representing the 'buildsPWDmatrix' section of the parameters.json parameter file.

    Attributes:
        folder (str): Output folder for the generated files. Default is 'tracing'.
        tracing_method (List[str]): List of available tracing methods. Default is ['masking', 'clustering'].
        mask_expansion (int): Maximum number of pixels to expand masks until they collide. Default is 8.
        masks2process (Dict[str, str]): Dictionary mapping mask names to their corresponding channels. Default is {'nuclei': 'DAPI', 'mask1': 'mask0'}.
        flux_min (int): Minimum flux required to keep an object. Default is 10.
        flux_min_3D (float): Minimum flux required to keep an object in 3D. Default is 0.1.
        KDtree_distance_threshold_mum (float): Distance threshold used to build KDtree. Default is 1.
        toleranceDrift (Union[int, List[int]]): ZXY tolerance used for block drift correction, in pixels. Default is [3, 1, 1].
        remove_uncorrected_localizations (bool): Flag indicating whether to remove uncorrected localizations. Default is True.
        z_offset (float): Z offset value. Default is 2.0.
        unknown_params (CatchAll): Catch-all field for any unknown parameters.

    """

    folder: str = set_default("folder", "tracing")
    tracing_method: List[str] = set_default("tracing_method", ["masking", "clustering"])
    mask_expansion: int = set_default("mask_expansion", 8)
    masks2process: Dict[str, str] = set_default(
        "masks2process", {"nuclei": "DAPI", "mask1": "mask0"}
    )
    flux_min: int = set_default("flux_min", 10)
    flux_min_3D: float = set_default("flux_min_3D", 0.1)
    KDtree_distance_threshold_mum: float = set_default("KDtree_distance_threshold_mum", 1)
    toleranceDrift: Union[int, List[int]] = set_default("toleranceDrift", [3, 1, 1])
    remove_uncorrected_localizations: bool = set_default(
        "remove_uncorrected_localizations", True
    )
    z_offset: float = set_default("z_offset", 2.0)
    unknown_params: CatchAll = field(default_factory=lambda: {})

    def __post_init__(self):
        if self.unknown_params:
            print(f"! Unknown parameters detected: {self.unknown_params}")
