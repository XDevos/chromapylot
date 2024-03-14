from dataclasses import dataclass, field
from typing import Dict, List, Union

from dataclasses_json import CatchAll, Undefined, dataclass_json, LetterCase
from chromapylot.parameters.utils import set_default


@dataclass_json(undefined=Undefined.INCLUDE, letter_case=LetterCase.CAMEL)
@dataclass
class ProjectionParams:
    """
    Represents the zProject section of the parameters.json parameter file.

    Attributes:
        folder (str): Output folder for the projected images.
        mode (str): Projection mode. Options: full, manual, automatic, laplacian.
        block_size (int): Block size for projection.
        display (bool): Flag indicating whether to display the projected images.
        zmin (int): Minimum z-slice to include in the projection.
        zmax (int): Maximum z-slice to include in the projection.
        zwindows (int): Number of z-slices to include in each projection window.
        window_security (int): Number of additional z-slices to include for security in each projection window.
        z_project_option (str): Projection option. Options: sum, MIP.
        unknown_params (Dict[str, Any]): Dictionary to store unknown parameters.
    """

    # pylint: disable=invalid-name
    folder: str = set_default("folder", "project")
    mode: str = set_default("mode", "full")
    block_size: int = set_default("block_size", 256)
    display: bool = set_default("display", True)
    zmin: int = set_default("zmin", 1)
    zmax: int = set_default("zmax", 59)
    zwindows: int = set_default("zwindows", 15)
    window_security: int = set_default("window_security", 2)
    z_project_option: str = set_default("z_project_option", "MIP")
    unknown_params: CatchAll = field(default_factory=lambda: {})

    def __post_init__(self):
        if self.unknown_params:
            print(f"! Unknown parameters detected: {self.unknown_params}")
