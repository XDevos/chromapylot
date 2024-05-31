from dataclasses import dataclass, field
from typing import Dict, List, Union

from dataclasses_json import CatchAll, Undefined, dataclass_json
from chromapylot.parameters.utils import set_default, warn_pop


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
    reducePlanes: bool = set_default("reducePlanes", False)
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
    limit_x: int = set_default("limit_x", 351)
    unknown_params: CatchAll = field(
        default_factory=lambda: {}
    )  # Catch-all field for unknown parameters

    def __post_init__(self):
        # Handle default values for certain parameters
        self._3Dmethod = (
            warn_pop(self.unknown_params, "3Dmethod", "None")
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
