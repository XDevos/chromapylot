from dataclasses import dataclass, field
from typing import Dict, List, Union

from dataclasses_json import CatchAll, Undefined, dataclass_json
from chromapylot.parameters.utils import set_default, warn_pop


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class RegistrationParams:
    """Represents the alignImages section of the parameters.json parameter file.

    Attributes:
        register_global_folder (str): Output folder for global registration.
        register_local_folder (str): Output folder for local registration.
        outputFile (str): Output file name for storing shifts.
        referenceFiducial (str): Reference fiducial for alignment.
        localAlignment (str): Type of local alignment. Options: None, mask2D, block3D.
        alignByBlock (bool): Flag indicating whether to perform block alignment.
        tolerance (float): Percentage of error tolerated in block alignment.
        lower_threshold (float): Lower threshold for adjusting image intensity levels before 2D alignment.
        higher_threshold (float): Higher threshold for adjusting image intensity levels before 2D alignment.
        _3D_lower_threshold (Union[float, str]): Lower threshold for adjusting image intensity levels before 3D alignment.
        _3D_higher_threshold (Union[float, str]): Higher threshold for adjusting image intensity levels before 3D alignment.
        background_sigma (float): Sigma value used to remove inhomogeneous background.
        blockSize (int): Block size for global registration.
        blockSizeXY (int): Block size for local registration in XY dimension.
        upsample_factor (int): Upsample factor for local registration.
        unknown_params (Dict[str, Any]): Dictionary to store unknown parameters.

    Note:
        - The `_3D_lower_threshold` and `_3D_higher_threshold` attributes can be set to a float value or "None".
    """

    # pylint: disable=invalid-name
    register_global_folder: str = set_default(
        "register_global_folder", "register_global"
    )
    register_local_folder: str = set_default("register_local_folder", "register_local")
    outputFile: str = set_default("outputFile", "shifts")
    referenceFiducial: str = set_default("referenceFiducial", "RT27")
    localAlignment: str = set_default("localAlignment", "block3D")
    alignByBlock: bool = set_default("alignByBlock", True)
    tolerance: float = set_default("tolerance", 0.1)
    lower_threshold: float = set_default("lower_threshold", 0.999)
    higher_threshold: float = set_default("higher_threshold", 0.9999999)
    _3D_lower_threshold: Union[float, str] = set_default("_3D_lower_threshold", "None")
    _3D_higher_threshold: Union[float, str] = set_default(
        "_3D_higher_threshold", "None"
    )
    background_sigma: float = set_default("background_sigma", 3.0)
    blockSize: int = set_default("blockSize", 256)
    blockSizeXY: int = set_default("blockSizeXY", 128)
    upsample_factor: int = set_default("upsample_factor", 100)
    unknown_params: CatchAll = field(default_factory=lambda: {})

    def __post_init__(self):
        self._3D_lower_threshold = (
            warn_pop(self.unknown_params, "3D_lower_threshold", 0.9)
            if self._3D_lower_threshold == "None"
            else self._3D_lower_threshold
        )
        self._3D_higher_threshold = (
            warn_pop(self.unknown_params, "3D_higher_threshold", 0.9999)
            if self._3D_higher_threshold == "None"
            else self._3D_higher_threshold
        )

        if self.unknown_params:  # if dict isn't empty
            print(f"! Unknown parameters detected: {self.unknown_params}")
