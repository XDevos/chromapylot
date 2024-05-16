#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import uuid
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

from modules.module import Module
from chromapylot.core.core_types import DataType
from chromapylot.core.data_manager import (
    DataManager,
    get_roi_number_from_image_path,
    create_png_path,
)
from chromapylot.parameters.segmentation_params import SegmentationParams
from chromapylot.parameters.projection_params import ProjectionParams
from chromapylot.parameters.acquisition_params import AcquisitionParams
from chromapylot.modules.project import (
    get_focus_plane,
    split_in_blocks,
    calculate_focus_per_block,
)

from astropy.visualization import simple_norm
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.table import Column, Table, vstack
from photutils import Background2D, DAOStarFinder, MedianBackground
from dask.distributed import Lock


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
        self.dirname = "localize_2d"
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
        return sources

    def save_data(self, data, input_path, input_data):
        self._save_localization_table(data, input_path)
        png_path = create_png_path(
            input_path, self.data_m.output_folder, self.dirname, "_segmentedSources"
        )
        self.show_image_sources(
            input_data,
            data,
            png_path,
        )

    def _save_localization_table(self, data, input_path):
        barcode_id = self.data_m.get_barcode_id(input_path)
        roi = get_roi_number_from_image_path(input_path)
        data = self.__update_localization_table(data, barcode_id, roi)
        out_path = os.path.join(
            self.data_m.output_folder,
            self.dirname,
            "data",
            "segmentedObjects_barcode.dat",
        )
        try:
            with Lock(out_path):
                self.__save_localization_table(data, out_path)
        except RuntimeError:
            self.__save_localization_table(data, out_path)

    def __update_localization_table(self, sources, barcode_id, roi):
        n_sources = len(sources)
        # buid
        buid = [str(uuid.uuid4()) for _ in range(n_sources)]
        col_buid = Column(buid, name="Buid", dtype=str)

        # barcode_id, cellID and roi
        col_roi = Column(int(roi) * np.ones(n_sources), name="ROI #", dtype=int)
        col_barcode = Column(
            int(barcode_id) * np.ones(n_sources), name="Barcode #", dtype=int
        )
        col_cell_id = Column(np.zeros(n_sources), name="CellID #", dtype=int)
        zcoord = Column(np.nan * np.zeros(n_sources), name="zcentroid", dtype=float)

        # adds to table
        sources.add_column(col_barcode, index=0)
        sources.add_column(col_roi, index=0)
        sources.add_column(col_buid, index=0)
        sources.add_column(col_cell_id, index=2)
        sources.add_column(zcoord, index=5)

        return sources

    def __save_localization_table(self, data, out_path):
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        elif os.path.exists(out_path):
            existing_table = Table.read(out_path, format="ascii.ecsv")
            data = vstack([existing_table, data])
        data.write(out_path, format="ascii.ecsv", overwrite=True)

    def show_image_sources(self, im, sources, output_filename):
        # estimates and removes inhomogeneous background
        bkg_estimator = MedianBackground()
        bkg = Background2D(
            im,
            (64, 64),
            filter_size=(3, 3),
            sigma_clip=SigmaClip(sigma=self.background_sigma),
            bkg_estimator=bkg_estimator,
        )
        im1_bkg_substracted = im - bkg.background

        percent = 99.5
        flux = sources["flux"]
        x = sources["xcentroid"] + 0.5
        y = sources["ycentroid"] + 0.5

        fig, ax = plt.subplots()
        fig.set_size_inches((50, 50))

        norm = simple_norm(im, "sqrt", percent=percent)
        ax.imshow(im1_bkg_substracted, cmap="Greys", origin="lower", norm=norm)
        p_1 = ax.scatter(
            x,
            y,
            c=flux,
            s=50,
            facecolors="none",
            cmap="jet",
            marker="x",
            vmin=0,
            vmax=2000,
        )
        fig.colorbar(p_1, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlim(0, im.shape[1] - 1)
        ax.set_ylim(0, im.shape[0] - 1)
        fig.savefig(output_filename)
        plt.close(fig)


class ReducePlanes(Module):
    def __init__(
        self,
        data_manager: DataManager,
        projection_params: ProjectionParams,
        acquisition_params: AcquisitionParams,
    ):
        super().__init__(
            data_manager=data_manager,
            input_type=DataType.IMAGE_3D,
            output_type=DataType.IMAGE_3D,
            reference_type=None,
            supplementary_type=None,
        )
        self.dirname = "reduce_planes"
        self.block_size_xy = projection_params.block_size
        self.z_window = int(projection_params.zwindows / acquisition_params.zBinning)

    def load_data(self, input_path):
        return self.data_m.load_image_3d(input_path)

    def run(self, data, supplementary_data=None):
        blocks = split_in_blocks(data, block_size_xy=self.block_size)
        focal_plane_matrix = calculate_focus_per_block(blocks)
        focus_plane = get_focus_plane(focal_plane_matrix)
        zmin = np.max([focus_plane - self.z_window, 0])
        zmax = np.min([focus_plane + self.z_window, data.shape[0]])
        z_range = range(zmin, zmax)
        reduce_img = np.empty(data.shape)
        reduce_img.fill(np.nan)
        reduce_img[z_range, :, :] = data[z_range, :, :]  # keep only the focus planes
        return reduce_img

    def save_data(self, data, input_path, input_data):
        pass


class Localize3D(Module):
    def __init__(
        self,
        data_manager: DataManager,
        segmentation_params: SegmentationParams,
    ):
        super().__init__(
            data_manager=data_manager,
            input_type=DataType.IMAGE_3D_SHIFTED,
            output_type=DataType.TABLE_3D,
            reference_type=None,
            supplementary_type=None,
        )
        self.dirname = "localize_3d"

    def load_data(self, input_path):
        return self.data_m.load_image_3d(input_path)

    def run(self, data, supplementary_data=None):
        # segments 3D volumes
        _, segmented_image_3d = self._segment_3d_volumes(image_3d_aligned)
