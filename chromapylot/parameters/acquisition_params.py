from dataclasses import dataclass, field
from typing import Dict, List, Union

from dataclasses_json import CatchAll, Undefined, dataclass_json
from chromapylot.parameters.utils import set_default


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class AcquisitionParams:
    """
    acquisition section of parameters.json parameter file.

    Attributes:
        DAPI_channel (str): The DAPI channel used for acquisition.
        RNA_channel (str): The RNA channel used for acquisition.
        barcode_channel (str): The barcode channel used for acquisition.
        mask_channel (str): The mask channel used for acquisition.
        fiducialBarcode_channel (str): The fiducial barcode channel used for acquisition.
        fiducialMask_channel (str): The fiducial mask channel used for acquisition.
        fiducialDAPI_channel (str): The fiducial DAPI channel used for acquisition.
        fileNameRegExp (str): The regular expression pattern for matching the file names.
        pixelSizeXY (float): The pixel size in the XY plane.
        pixelSizeZ (float): The pixel size in the Z direction.
        zBinning (int): The Z binning factor.
        unknown_params (CatchAll): Catch-all field for unknown parameters.
    """

    # pylint: disable=invalid-name
    DAPI_channel: str = set_default("DAPI_channel", "None")
    RNA_channel: str = set_default("RNA_channel", "None")
    barcode_channel: str = set_default("barcode_channel", "None")
    mask_channel: str = set_default("mask_channel", "None")
    fiducialBarcode_channel: str = set_default("fiducialBarcode_channel", "None")
    fiducialMask_channel: str = set_default("fiducialMask_channel", "None")
    fiducialDAPI_channel: str = set_default("fiducialDAPI_channel", "None")
    fileNameRegExp: str = set_default(
        "fileNameRegExp",
        "scan_(?P<runNumber>[0-9]+)_(?P<cycle>[\\w|-]+)_(?P<roi>[0-9]+)_ROI_converted_decon_(?P<channel>[\\w|-]+).tif",
    )
    pixelSizeXY: float = set_default("pixelSizeXY", 0.1)
    pixelSizeZ: float = set_default("pixelSizeZ", 0.25)
    zBinning: int = set_default("zBinning", 2)
    unknown_params: CatchAll = field(default_factory=lambda: {})

    def __post_init__(self):
        if self.unknown_params:
            print(f"! Unknown parameters detected: {self.unknown_params}")
