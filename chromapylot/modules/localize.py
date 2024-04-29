#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
from typing import Dict, List
import numpy as np

from modules.module import Module
from chromapylot.core.core_types import DataType
from chromapylot.core.data_manager import DataManager
from chromapylot.parameters.segmentation_params import SegmentationParams

from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.table import Column, Table, vstack
from photutils import Background2D, DAOStarFinder, MedianBackground


class Localize2D(Module):
    def __init__(
        self,
        data_manager: DataManager,
        segmentation_params: SegmentationParams,
    ):
        super().__init__(
            data_manager=data_manager,
            input_type=DataType.IMAGE_2D_SHIFTED,
            output_type=DataType.TABLE_2D,
            reference_type=None,
            supplementary_type=None,
        )
        self.background_method = segmentation_params.background_method
        self.background_sigma = segmentation_params.background_sigma
        self.threshold_over_std = segmentation_params.threshold_over_std
        self.fwhm = segmentation_params.fwhm
        self.brightest = segmentation_params.brightest

    def load_data(self, input_path):
        return self.data_m.load_image_2d(input_path)

    def run(self, data, supplementary_data=None):
        if self.background_method != "inhomogeneous":
            raise ValueError(
                f"Segmentation method {self.background_method} not recognized, only 'inhomogeneous' is supported for localize_2d"
            )

        sigma_clip = SigmaClip(sigma=self.background_sigma)

        # estimates and removes inhomogeneous background
        bkg_estimator = MedianBackground()
        bkg = Background2D(
            data,
            (64, 64),
            filter_size=(3, 3),
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
        )
        im1_bkg_substracted = data - bkg.background
        _, _, std = sigma_clipped_stats(im1_bkg_substracted, sigma=3.0)

        # estimates sources
        daofind = DAOStarFinder(
            fwhm=self.fwhm,
            threshold=self.threshold_over_std * std,
            brightest=self.brightest,
            exclude_border=True,
        )
        sources = daofind(im1_bkg_substracted)
        localization_table = self.create_localization_table(sources)
        return localization_table

    def save_data(self, data, input_path, input_data):
        raise NotImplementedError

    def create_localization_table(self, sources):

        # buid
        buid = [str(uuid.uuid4()) for _ in range(len(output))]
        col_buid = Column(buid, name="Buid", dtype=str)

        # barcode_id, cellID and roi
        barcode_id = os.path.basename(file_name).split("_")[2].split("RT")[1]
        col_roi = Column(int(roi) * np.ones(len(output)), name="ROI #", dtype=int)
        col_barcode = Column(
            int(barcode_id) * np.ones(len(output)), name="Barcode #", dtype=int
        )
        col_cell_id = Column(np.zeros(len(output)), name="CellID #", dtype=int)
        zcoord = Column(np.nan * np.zeros(len(output)), name="zcentroid", dtype=float)

        if output[0] is not None:
            # adds to table
            output.add_column(col_barcode, index=0)
            output.add_column(col_roi, index=0)
            output.add_column(col_buid, index=0)
            output.add_column(col_cell_id, index=2)
            output.add_column(zcoord, index=5)
