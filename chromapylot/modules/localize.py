#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import uuid
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops

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
from chromapylot.core.data_manager import get_file_path
from astropy.visualization import simple_norm
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.table import Column, Table, vstack
from photutils import Background2D, DAOStarFinder, MedianBackground
from dask.distributed import Lock
from astropy.table import Table
from skimage import exposure
from apifish.detection.spot_modeling import fit_subpixel


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
            output_type=DataType.REDUCE_TUPLE,
            reference_type=None,
            supplementary_type=None,
        )
        self.dirname = "reduce_planes"
        self.block_size = projection_params.block_size
        self.z_window = int(projection_params.zwindows / acquisition_params.zBinning)

    def load_data(self, input_path):
        return self.data_m.load_image_3d(input_path)

    def run(self, data, supplementary_data=None):
        blocks = split_in_blocks(data, block_size_xy=self.block_size)
        focal_plane_matrix = calculate_focus_per_block(blocks)
        focus_plane = get_focus_plane(focal_plane_matrix)
        zmin = np.max([focus_plane - self.z_window, 0])
        zmax = np.min([focus_plane + self.z_window, data.shape[0]])
        # reduce_img = np.empty(data.shape)
        # reduce_img.fill(np.nan)
        # reduce_img[z_range, :, :] = data[z_range, :, :]  # keep only the focus planes
        return (zmin, zmax)

    def save_data(self, data, input_path, input_data):
        pass


class ExtractProperties(Module):
    def __init__(
        self, data_manager: DataManager, segmentation_params: SegmentationParams
    ):
        super().__init__(
            data_manager=data_manager,
            input_type=DataType.SEGMENTED_3D,
            output_type=DataType.TABLE_3D,
            supplementary_type=[
                DataType.IMAGE_3D_SHIFTED,
                DataType.IMAGE_3D,
            ],
        )
        self.dirname = "localize_3d"
        self.brightest = segmentation_params.brightest
        self.threshold = segmentation_params._3D_threshold_over_std

    def run(self, image, mask):
        # gets object properties
        properties = regionprops(mask, intensity_image=image)
        # selects n_tolerance brightest spots and keeps only these for further processing
        try:
            # compatibility with scikit_image versions <= 0.18
            peak0 = [x.max_intensity for x in properties]
        except AttributeError:
            # compatibility with scikit_image versions >=0.19
            peak0 = [x.intensity_max for x in properties]
        peak_list = peak0.copy()
        peak_list.sort()
        if self.brightest == "None":
            last2keep = len(peak_list)
        else:
            last2keep = np.min([self.brightest, len(peak_list)])
        highest_peak_value = peak_list[-last2keep]
        selection = list(np.nonzero(peak0 > highest_peak_value)[0])
        # attributes properties using the brightests spots selected
        try:
            # compatibility with scikit_image versions <= 0.18
            peak = [properties[x].max_intensity for x in selection]
            centroids = [properties[x].weighted_centroid for x in selection]
            sharpness = [
                float(properties[x].filled_area / properties[x].bbox_area)
                for x in selection
            ]
            roundness1 = [properties[x].equivalent_diameter for x in selection]
        except AttributeError:
            # compatibility with scikit_image versions >=0.19
            peak = [properties[x].intensity_max for x in selection]
            centroids = [properties[x].centroid_weighted for x in selection]
            sharpness = [
                float(properties[x].area_filled / properties[x].area_bbox)
                for x in selection
            ]
            roundness1 = [properties[x].equivalent_diameter_area for x in selection]
        roundness2 = [properties[x].extent for x in selection]
        npix = [properties[x].area for x in selection]
        sky = [0.0 for x in selection]
        try:
            # compatibility with scikit_image versions <= 0.18
            peak = [properties[x].max_intensity for x in selection]
            flux = [
                100 * properties[x].max_intensity / self.threshold for x in selection
            ]  # peak intensity over t$
        except AttributeError:
            # compatibility with scikit_image versions >=0.19
            peak = [properties[x].intensity_max for x in selection]
            flux = [
                100 * properties[x].intensity_max / self.threshold for x in selection
            ]  # peak intensity$
        mag = [-2.5 * np.log10(x) for x in flux]  # -2.5 log10(flux)
        # converts centroids to spot coordinates for bigfish to run 3D gaussian fits
        z = [x[0] for x in centroids]
        y = [x[1] for x in centroids]
        x = [x[2] for x in centroids]
        spots = np.zeros((len(z), 3))
        spots[:, 0] = z
        spots[:, 1] = y
        spots[:, 2] = x
        spots = spots.astype("int64")
        return (
            spots,
            sharpness,
            roundness1,
            roundness2,
            npix,
            sky,
            peak,
            flux,
            mag,
        )

    def save_data(self, data, input_path, input_data):
        data.write(
            get_file_path(self.data_m.output_folder, "_props", "ecsv"),
            format="ascii.ecsv",
            overwrite=True,
        )


class FitSubpixel(Module):
    def __init__(
        self, data_manager: DataManager, segmentation_params: SegmentationParams
    ):
        super().__init__(
            data_manager=data_manager,
            input_type=DataType.IMAGE_3D,
            output_type=DataType.TABLE_3D,
            supplementary_type=DataType.TABLE_3D,
        )

    def run(self, image, properties):
        print("> Refits spots using gaussian 3D fittings...")

        print(" > Rescales image values after reinterpolation")
        image_3d_aligned = exposure.rescale_intensity(
            image_3d_aligned, out_range=(0, 1)
        )  # removes negative intensity levels

        # calls bigfish to get 3D sub-pixel coordinates based on 3D gaussian fitting
        # compatibility with latest version of bigfish. To be removed if stable.
        # TODO: Is it stable ? I think we can remove it.
        try:
            # version 0.4 commit fa0df4f
            spots_subpixel = fit_subpixel(
                image_3d_aligned,
                spots,
                voxel_size_z=p["voxel_size_z"],
                voxel_size_yx=p["voxel_size_yx"],
                psf_z=p["psf_z"],
                psf_yx=p["psf_yx"],
            )
        except TypeError:
            # version > 0.5
            spots_subpixel = fit_subpixel(
                image_3d_aligned,
                spots,
                (
                    p["voxel_size_z"],
                    p["voxel_size_yx"],
                    p["voxel_size_yx"],
                ),  # voxel size
                (p["psf_z"], p["psf_yx"], p["psf_yx"]),
            )  # spot radius

        print(" > Updating table and saving results")
        # updates table
        for i in range(spots_subpixel.shape[0]):
            z, x, y = spots_subpixel[i, :]
            table_entry = [
                str(uuid.uuid4()),
                roi,
                0,
                int(label.split("RT")[1]),
                i,
                z + z_offset,
                y,
                x,
                sharpness[i],
                roundness1[i],
                roundness2[i],
                npix[i],
                sky[i],
                peak[i],
                flux[i],
                mag[i],
            ]
            output_table.add_row(table_entry)

    def save_data(self, data, input_path, input_data):
        data.write(
            get_file_path(self.data_m.output_folder, "_fitted", "ecsv"),
            format="ascii.ecsv",
            overwrite=True,
        )
