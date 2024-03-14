#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes and functions to manage parameters.
"""

import copy
import json
import os
from dataclasses import asdict, dataclass, field
from os import path
from typing import Dict, List, Union

from dataclasses_json import CatchAll, LetterCase, Undefined, dataclass_json

from chromapylot.core.core_types import AnalysisType


def print_section(section: str):
    print(f"$ Load: {section}")


def load_json(file_name):
    """Load a JSON file like a python dict

    Parameters
    ----------
    file_name : str
        JSON file name

    Returns
    -------
    dict
        Python dict
    """
    if os.path.exists(file_name):
        with open(file_name, encoding="utf-8") as json_file:
            return json.load(json_file)
    return None


def warn_default(key, val):
    if val != "None":
        print(
            f"""! key NOT FOUND inside parameters.json: "{key}"\n\t\t  Default value used: {val}"""
        )
    return val


def warn_pop(dico: dict, key: str, default):
    if dico.get(key):
        return dico.pop(key, default)
    return warn_default(key, default)


def set_default(key: str, val):
    # pylint: disable=invalid-field-call
    return field(default_factory=lambda: warn_default(key, val))


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


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class SegmentationParams:
    """Represents the segmentedObjects section of the parameters.json parameter file.

    Attributes:
        mask_2d_folder (str): Output folder for 2D masks.
        mask_3d_folder (str): Output folder for 3D masks.
        localize_2d_folder (str): Output folder for 2D localizations.
        localize_3d_folder (str): Output folder for 3D localizations.
        operation (str): Options for segmentation operation (2D, 3D).
        outputFile (str): Output file name for localizations.
        background_method (str): Method for background removal (flat, inhomogeneous, stardist).
        stardist_basename (str): Base name for stardist network.
        stardist_network (str): Network for 2D mask segmentation.
        stardist_network3D (str): Network for 3D mask or barcode segmentation.
        tesselation (bool): Whether to tesselate masks.
        background_sigma (float): Sigma value used to remove inhomogeneous background.
        threshold_over_std (float): Threshold value used to detect sources.
        fwhm (float): Source size in pixels (FWHM).
        brightest (int): Maximum number of sources segmented per field of view.
        intensity_min (int): Minimum intensity to keep an object.
        intensity_max (int): Maximum intensity to keep an object.
        area_min (int): Minimum area to keep an object.
        area_max (int): Maximum area to keep an object.
        reducePlanes (bool): Whether to reduce planes for 3D segmentation.
        residual_max (float): Maximum residuals to keep an object in z-profile Fit.
        sigma_max (int): Maximum sigma value for 3D fitting to keep an object in z-profile Fit.
        centroidDifference_max (int): Maximum difference in centroid position to associate a source localized in YZ with one localized in XY.
        _3Dmethod (str): Method for 3D segmentation: thresholding or stardist.
        _3DGaussianfitWindow (Union[int, str]): Window size to extract subVolume for z-profile Fit.
        _3dAP_window (Union[int, str]): Window size for constructing a YZ image by summing from xPlane-window to xPlane+window.
        _3dAP_flux_min (Union[int, str]): Threshold to keep a source detected in YZ.
        _3dAP_brightest (Union[int, str]): Number of sources sought in each YZ plane.
        _3dAP_distTolerance (Union[int, str]): Pixel distance to attribute a source localized in YZ to one localized in XY.
        _3D_threshold_over_std (Union[int, str]): Threshold value used for 3D segmentation.
        _3D_sigma (Union[int, str]): Sigma value used for 3D segmentation.
        _3D_boxSize (Union[int, str]): Box size used for 3D segmentation.
        _3D_area_min (Union[int, str]): Minimum area to keep an object in 3D segmentation.
        _3D_area_max (Union[int, str]): Maximum area to keep an object in 3D segmentation.
        _3D_nlevels (Union[int, str]): Number of levels used for 3D segmentation.
        _3D_contrast (Union[float, str]): Contrast value used for 3D segmentation.
        _3D_psf_z (Union[int, str]): PSF value in z-direction used for 3D segmentation.
        _3D_psf_yx (Union[int, str]): PSF value in yx-direction used for 3D segmentation.
        _3D_lower_threshold (Union[float, str]): Lower threshold value used for 3D segmentation.
        _3D_higher_threshold (Union[float, str]): Higher threshold value used for 3D segmentation.
        unknown_params (CatchAll): Catch-all field for unknown parameters.
    """

    # pylint: disable=invalid-name
    mask_2d_folder: str = set_default("mask_2d_folder", "mask_2d")
    mask_3d_folder: str = set_default("mask_3d_folder", "mask_3d")
    localize_2d_folder: str = set_default("localize_2d_folder", "localize_2d")
    localize_3d_folder: str = set_default("localize_3d_folder", "localize_3d")
    operation: str = set_default("operation", "2D,3D")
    outputFile: str = set_default("outputFile", "localizations")
    background_method: str = set_default("background_method", "inhomogeneous")
    stardist_basename: str = set_default("stardist_basename", "None")
    stardist_network: str = set_default("stardist_network", "None")
    stardist_network3D: str = set_default("stardist_network3D", "None")
    tesselation: bool = set_default("tesselation", True)
    background_sigma: float = set_default("background_sigma", 3.0)
    threshold_over_std: float = set_default("threshold_over_std", 1.0)
    fwhm: float = set_default("fwhm", 3.0)
    brightest: int = set_default("brightest", 1100)
    intensity_min: int = set_default("intensity_min", 0)
    intensity_max: int = set_default("intensity_max", 59)
    area_min: int = set_default("area_min", 50)
    area_max: int = set_default("area_max", 500)
    reducePlanes: bool = set_default("reducePlanes", True)
    residual_max: float = set_default("residual_max", 2.5)
    sigma_max: int = set_default("sigma_max", 5)
    centroidDifference_max: int = set_default("centroidDifference_max", 5)
    _3Dmethod: str = set_default("_3Dmethod", "None")
    _3DGaussianfitWindow: Union[int, str] = set_default("_3DGaussianfitWindow", "None")
    _3dAP_window: Union[int, str] = set_default("_3dAP_window", "None")
    _3dAP_flux_min: Union[int, str] = set_default("_3dAP_flux_min", "None")
    _3dAP_brightest: Union[int, str] = set_default("_3dAP_brightest", "None")
    _3dAP_distTolerance: Union[int, str] = set_default("_3dAP_distTolerance", "None")
    _3D_threshold_over_std: Union[int, str] = set_default(
        "_3D_threshold_over_std", "None"
    )
    _3D_sigma: Union[int, str] = set_default("_3D_sigma", "None")
    _3D_boxSize: Union[int, str] = set_default("_3D_boxSize", "None")
    _3D_area_min: Union[int, str] = set_default("_3D_area_min", "None")
    _3D_area_max: Union[int, str] = set_default("_3D_area_max", "None")
    _3D_nlevels: Union[int, str] = set_default("_3D_nlevels", "None")
    _3D_contrast: Union[float, str] = set_default("_3D_contrast", "None")
    _3D_psf_z: Union[int, str] = set_default("_3D_psf_z", "None")
    _3D_psf_yx: Union[int, str] = set_default("_3D_psf_yx", "None")
    _3D_lower_threshold: Union[float, str] = set_default("_3D_lower_threshold", "None")
    _3D_higher_threshold: Union[float, str] = set_default(
        "_3D_higher_threshold", "None"
    )
    unknown_params: CatchAll = field(
        default_factory=lambda: {}
    )  # Catch-all field for unknown parameters

    def __post_init__(self):
        # Handle default values for certain parameters
        self._3Dmethod = (
            warn_pop(self.unknown_params, "3Dmethod", "stardist")
            if self._3Dmethod == "None"
            else self._3Dmethod
        )
        self._3DGaussianfitWindow = (
            warn_pop(self.unknown_params, "3DGaussianfitWindow", 3)
            if self._3DGaussianfitWindow == "None"
            else self._3DGaussianfitWindow
        )
        self._3dAP_window = (
            warn_pop(self.unknown_params, "3dAP_window", 5)
            if self._3dAP_window == "None"
            else self._3dAP_window
        )
        self._3dAP_flux_min = (
            warn_pop(self.unknown_params, "3dAP_flux_min", 2)
            if self._3dAP_flux_min == "None"
            else self._3dAP_flux_min
        )
        self._3dAP_brightest = (
            warn_pop(self.unknown_params, "3dAP_brightest", 100)
            if self._3dAP_brightest == "None"
            else self._3dAP_brightest
        )
        self._3dAP_distTolerance = (
            warn_pop(self.unknown_params, "3dAP_distTolerance", 1)
            if self._3dAP_distTolerance == "None"
            else self._3dAP_distTolerance
        )
        self._3D_threshold_over_std = (
            warn_pop(self.unknown_params, "3D_threshold_over_std", 5)
            if self._3D_threshold_over_std == "None"
            else self._3D_threshold_over_std
        )
        self._3D_sigma = (
            warn_pop(self.unknown_params, "3D_sigma", 3)
            if self._3D_sigma == "None"
            else self._3D_sigma
        )
        self._3D_boxSize = (
            warn_pop(self.unknown_params, "3D_boxSize", 32)
            if self._3D_boxSize == "None"
            else self._3D_boxSize
        )
        self._3D_area_min = (
            warn_pop(self.unknown_params, "3D_area_min", 10)
            if self._3D_area_min == "None"
            else self._3D_area_min
        )
        self._3D_area_max = (
            warn_pop(self.unknown_params, "3D_area_max", 250)
            if self._3D_area_max == "None"
            else self._3D_area_max
        )
        self._3D_nlevels = (
            warn_pop(self.unknown_params, "3D_nlevels", 64)
            if self._3D_nlevels == "None"
            else self._3D_nlevels
        )
        self._3D_contrast = (
            warn_pop(self.unknown_params, "3D_contrast", 0.001)
            if self._3D_contrast == "None"
            else self._3D_contrast
        )
        self._3D_psf_z = (
            warn_pop(self.unknown_params, "3D_psf_z", 500)
            if self._3D_psf_z == "None"
            else self._3D_psf_z
        )
        self._3D_psf_yx = (
            warn_pop(self.unknown_params, "3D_psf_yx", 200)
            if self._3D_psf_yx == "None"
            else self._3D_psf_yx
        )
        self._3D_lower_threshold = (
            warn_pop(self.unknown_params, "3D_lower_threshold", 0.99)
            if self._3D_lower_threshold == "None"
            else self._3D_lower_threshold
        )
        self._3D_higher_threshold = (
            warn_pop(self.unknown_params, "3D_higher_threshold", 0.9999)
            if self._3D_higher_threshold == "None"
            else self._3D_higher_threshold
        )
        if self.unknown_params:
            print(f"! Unknown parameters detected: {self.unknown_params}")


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
        KDtree_distance_threshold_mum (int): Distance threshold used to build KDtree. Default is 1.
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
    KDtree_distance_threshold_mum: int = set_default("KDtree_distance_threshold_mum", 1)
    toleranceDrift: Union[int, List[int]] = set_default("toleranceDrift", [3, 1, 1])
    remove_uncorrected_localizations: bool = set_default(
        "remove_uncorrected_localizations", True
    )
    z_offset: float = set_default("z_offset", 2.0)
    unknown_params: CatchAll = field(default_factory=lambda: {})

    def __post_init__(self):
        if self.unknown_params:
            print(f"! Unknown parameters detected: {self.unknown_params}")


class PipelineParams:
    def __init__(self, raw_params: Dict[str, Dict[str, dict]], label: AnalysisType):
        labelled_params = merge_common_and_labels(raw_params, label)
        acq_p = labelled_params.get("acquisition", {})
        prj_p = labelled_params.get("zProject", {})
        rgt_p = labelled_params.get("alignImages", {})
        sgm_p = labelled_params.get("segmentedObjects", {})
        mtx_p = labelled_params.get("buildsPWDmatrix", {})

        self.acquisition = AcquisitionParams.from_dict(acq_p) if acq_p else None
        self.projection = ProjectionParams.from_dict(prj_p) if prj_p else None
        self.registration = RegistrationParams.from_dict(rgt_p) if rgt_p else None
        self.segmentation = SegmentationParams.from_dict(sgm_p) if sgm_p else None
        self.matrix = MatrixParams.from_dict(mtx_p) if mtx_p else None

        self.highlight_deprecated_params(labelled_params)

    def highlight_deprecated_params(self, dict_to_check: dict):
        """Warns the user that there are unused/deprecated parameters in his parameters.json

        Parameters
        ----------
        dict_to_check : dict
            _description_
        """
        for key in dict_to_check:
            if key not in [
                "acquisition",
                "zProject",
                "alignImages",
                "segmentedObjects",
                "buildsPWDmatrix",
            ]:
                unused_params = {key: dict_to_check[key]}
                print(
                    f"! Unused parameters detected, it's probably a deprecated section: {unused_params}"
                )

    def print_as_dict(self):
        result = {}
        attr_to_print = [
            (self.acquisition, "acquisition"),
            (self.projection, "zProject"),
            (self.registration, "alignImages"),
            (self.segmentation, "segmentedObjects"),
            (self.matrix, "buildsPWDmatrix"),
        ]
        for attribute, key in attr_to_print:
            if not (attribute is None or attribute == "None"):
                result[key] = asdict(attribute)
                # remove "unknown_params" section
                result[key].pop("unknown_params", None)
        return result

    def get_module_params(self, module_name: str):
        if module_name in ["acquisition", "skip"]:
            return {"acquisition_params": self.acquisition}
        if module_name == "project":
            return {"projection_params": self.projection}
        if module_name in ["register_global", "register_local", "shift_2d", "shift_3d"]:
            return {"registration_params": self.registration}
        if module_name in [
            "segment_2d",
            "segment_3d",
            "extract_2d",
            "extract_3d",
            "filter_mask",
            "select_mask_2d",
            "select_mask_3d",
        ]:
            return {"segmentation_params": self.segmentation}
        if module_name in [
            "filter_localization",
            "register_localization",
            "build_matrix",
        ]:
            return {"matrix_params": self.matrix}
        if module_name in ["build_trace_3d", "build_trace_2d"]:
            return {
                "acquisition_params": self.acquisition,
                "matrix_params": self.matrix,
            }
        raise ValueError(f"Unknown module name: {module_name}")


def deep_dict_update(main_dict: dict, new_dict: dict):
    """Update recursively a nested dict with another.
    main_dict keys/values that do not exist in new_dict are kept.

    Parameters
    ----------
    main_dict : dict
        Main dict with all default values
    new_dict : dict
        Dict with new values to update

    Returns
    -------
    dict
        The main_dict overwrite by new_dict value
    """
    main_deep_copy = copy.deepcopy(main_dict)
    for key, value in new_dict.items():
        if isinstance(value, dict):
            main_deep_copy[key] = deep_dict_update(main_deep_copy.get(key, {}), value)
        else:
            main_deep_copy[key] = value
    return main_deep_copy


def merge_common_and_labels(raw_params, label):
    common_params = raw_params["common"]
    if label.value not in raw_params["labels"]:
        return common_params
    label_params = raw_params["labels"][label.value]
    return deep_dict_update(common_params, label_params)
