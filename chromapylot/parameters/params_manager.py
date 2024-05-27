#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes and functions to manage parameters.
"""

from dataclasses import asdict
from os import path
from typing import Dict

from chromapylot.core.core_types import AnalysisType, RoutineName
from chromapylot.parameters.utils import merge_common_and_labels
from chromapylot.parameters.acquisition_params import AcquisitionParams
from chromapylot.parameters.projection_params import ProjectionParams
from chromapylot.parameters.matrix_params import MatrixParams
from chromapylot.parameters.registration_params import RegistrationParams
from chromapylot.parameters.segmentation_params import SegmentationParams


class ParamsManager:
    def __init__(self, raw_params: Dict[str, Dict[str, dict]], label: AnalysisType):
        labelled_params = merge_common_and_labels(raw_params, label)
        acq_p = labelled_params.get("acquisition", {})
        prj_p = labelled_params.get("zProject", {})
        rgt_p = labelled_params.get("alignImages", {})
        sgm_p = labelled_params.get("segmentedObjects", {})
        mtx_p = labelled_params.get("buildsPWDmatrix", {})

        self.acquisition = AcquisitionParams.from_dict(acq_p)
        self.projection = ProjectionParams.from_dict(prj_p)
        self.registration = RegistrationParams.from_dict(rgt_p)
        self.segmentation = SegmentationParams.from_dict(sgm_p)
        self.matrix = MatrixParams.from_dict(mtx_p)

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

    def get_module_params(self, routine_name: RoutineName):
        module_name = routine_name.value
        if module_name in ["acquisition", "skip"]:
            return {"acquisition_params": self.acquisition}
        if module_name in [
            "project",
            "project_by_block",
            "interpolate_focal_plane",
            "split_in_blocks",
        ]:
            return {"projection_params": self.projection}
        if module_name in [
            "register_global",
            "register_by_block",
            "compare_block_global",
            "register_local",
            "shift_2d",
            "shift_3d",
        ]:
            return {"registration_params": self.registration}
        if module_name in [
            "localize_2d",
            "segment_2d",
            "segment_3d",
            "deblend_3d",
            "extract_2d",
            "extract_properties",
            "filter_mask",
            "select_mask_2d",
            "select_mask_3d",
            "fit_subpixel",
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
        if module_name in ["reduce_planes"]:
            return {
                "acquisition_params": self.acquisition,
                "projection_params": self.projection,
            }
        if module_name in ["preprocess_3d"]:
            return {
                "registration_params": self.registration,
                "segmentation_params": self.segmentation,
            }
        if module_name in ["shift_spot_on_z"]:
            return {}
        raise ValueError(f"Unknown module name: {module_name}")
