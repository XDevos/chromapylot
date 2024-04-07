#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from typing import List
from chromapylot.modules.module import Module
from chromapylot.core.core_types import DataType
from chromapylot.parameters.registration_params import RegistrationParams
import numpy as np
import os
from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground
from skimage.exposure import match_histograms
from skimage.util.shape import view_as_blocks
from skimage.registration import phase_cross_correlation
from tqdm import trange
from scipy.ndimage import shift as shift_image
from numpy import linalg as LA
from chromapylot.core.data_manager import load_json
from chromapylot.core.data_manager import get_roi_number_from_image_path
from chromapylot.core.data_manager import DataManager
from datetime import datetime
from chromapylot.core.data_manager import save_npy

import matplotlib.pyplot as plt


class RegisterGlobalModule(Module):
    def __init__(self, registration_params: RegistrationParams):
        super().__init__(
            input_type=DataType.IMAGE_2D,
            output_type=DataType.SHIFT_TUPLE,
            reference_type=DataType.IMAGE_2D,
        )
        self.dirname = "register_global"
        self.reference_data = None
        self.ref_fiducial = registration_params.referenceFiducial
        self.background_sigma = registration_params.background_sigma
        self.align_by_block = registration_params.alignByBlock
        self.block_size = registration_params.blockSize
        self.tolerance = registration_params.tolerance

    def run(self, raw_2d_img):
        if raw_2d_img is None:
            print("> No need to align reference image.")
            return None
        print(f"[Run] Register Global")
        preprocessed_img = remove_inhomogeneous_background(
            raw_2d_img, self.background_sigma
        )
        if self.align_by_block:
            print("> Align by block")
            ref_img = np.float32(self.reference_data)
            raw_img = np.float32(
                match_histograms(np.float32(preprocessed_img), ref_img)
            )
            (
                shifts,
                shift_by_block,
            ) = align_images_by_blocks(
                ref_img,
                raw_img,
                self.block_size,
                tolerance=self.tolerance,
            )
            mask = get_rms_mask(rms_image, self.tolerance)
            masked_block_shifts = np.where(mask[..., None], shift_by_block, np.nan)
            relative_shifts = compute_relative_shifts(masked_block_shifts)
            return {
                "shifts": shifts,
                "block_shifts": relative_shifts,
                "rms_image": rms_image,
            }

        else:
            # align whole image
            print("> Align whole image")
            raise NotImplementedError("Whole image alignment is not implemented yet.")

    def load_data(self, input_path, in_dir_length):
        if self.ref_fiducial in os.path.basename(input_path):
            return None
        print(f"[Load] {self.input_type.value}")
        short_path = input_path[in_dir_length:]
        print(f"> $INPUT{short_path}")
        return np.load(input_path)

    def save_data(self, data, output_dir, input_path):
        if data is None:
            return
        print("Saving shift tuple.")
        self._save_shift_tuple(data["shifts"], output_dir, input_path)
        # self._save_table(data["shifts"], output_dir, input_path)
        # self._save_registered(data["shifts"], output_dir, input_path)
        # self._save_error_alignment_block_map(data["shifts"], output_dir, input_path)
        # self._save_rms_block_map(data["shifts"], output_dir, input_path)
        # self._save_block_alignments(data["shifts"], output_dir, input_path)
        # self._save_overlay_corrected(data["shifts"], output_dir, input_path)
        # self._save_reference_difference(data["shifts"], output_dir, input_path)
        # self._save_intensity_hist(data["shifts"], output_dir, input_path)

    def load_reference_data(self, paths: List[str]):
        good_path = None
        for path in paths:
            if self.ref_fiducial in os.path.basename(path):
                good_path = path
                break
        if good_path[-3:] == "npy":
            ref_img = np.load(good_path)
            self.reference_data = remove_inhomogeneous_background(
                ref_img, self.background_sigma
            )
        else:
            raise NotImplementedError("Reference data must be a 2D numpy file.")

    def _save_shift_tuple(self, shifts, output_dir, input_path):
        out_path = os.path.join(output_dir, self.dirname, "data", "shifts.json")
        cycle = DataManager.get_cycle_from_path(input_path)
        roi = get_roi_number_from_image_path(input_path)
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        if not os.path.exists(out_path):
            existing_dict = {f"ROI:{roi}": {}}
        else:
            existing_dict = load_json(out_path)
        existing_dict[f"ROI:{roi}"][cycle] = shifts.tolist()
        with open(out_path, "w") as file:
            json.dump(existing_dict, file, ensure_ascii=False, sort_keys=True, indent=4)


class RegisterByBlock(Module):
    def __init__(self, registration_params: RegistrationParams):
        super().__init__(
            input_type=DataType.IMAGE_2D,
            output_type=DataType.MATRIX_3D,
            reference_type=DataType.IMAGE_2D,
        )
        self.dirname = "register_global"
        self.reference_data = None
        self.ref_fiducial = registration_params.referenceFiducial
        self.background_sigma = registration_params.background_sigma
        self.align_by_block = registration_params.alignByBlock
        self.block_size = registration_params.blockSize
        self.tolerance = registration_params.tolerance

    def run(self, raw_2d_img):
        if raw_2d_img is None:
            print("> No need to align reference image.")
            return None
        if not self.align_by_block:
            raise ValueError("This module is only for block alignment.")
        print(f"[Run] Register Global (by block)")
        preprocessed_img = remove_inhomogeneous_background(
            raw_2d_img, self.background_sigma
        )
        ref_img = np.float32(self.reference_data)
        raw_img = np.float32(match_histograms(np.float32(preprocessed_img), ref_img))
        return compute_shifts_and_rms_by_block(ref_img, raw_img, self.block_size)

    def load_data(self, input_path, in_dir_length):
        if self.ref_fiducial in os.path.basename(input_path):
            return None
        print(f"[Load] {self.input_type.value}")
        short_path = input_path[in_dir_length:]
        print(f"> $INPUT{short_path}")
        return np.load(input_path)

    def save_data(self, data, output_dir, input_path):
        if data is None:
            return
        print("Saving block shifts.")
        self._save_rms_block_map(data, output_dir, input_path)
        self._save_error_alignment_block_map(data, output_dir, input_path)
        self._save_block_alignments(data, output_dir, input_path)

    def load_reference_data(self, paths: List[str]):
        good_path = None
        for path in paths:
            if self.ref_fiducial in os.path.basename(path):
                good_path = path
                break
        if good_path[-3:] == "npy":
            ref_img = np.load(good_path)
            self.reference_data = remove_inhomogeneous_background(
                ref_img, self.background_sigma
            )
        else:
            raise NotImplementedError("Reference data must be a 2D numpy file.")

    def _save_rms_block_map(self, shifts_and_rms, output_dir, input_path):
        print(input_path)
        print(shifts_and_rms[:, :, 2])
        print(output_dir)
        base = os.path.basename(input_path).split(".")[0]
        npy_filename = base + "_rms_block_map.npy"
        out_path = os.path.join(output_dir, self.dirname, "data", npy_filename)
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        np.save(out_path, shifts_and_rms[:, :, 2])
        short_path = input_path[len(output_dir) :]
        print(f"> $OUTPUT{short_path}")

    def _save_error_alignment_block_map(self, shifts_and_rms, output_dir, input_path):
        # out_path = os.path.join(output_dir, self.dirname, "data", "error_alignment_block_map.npy")
        # if not os.path.exists(os.path.dirname(out_path)):
        #     os.makedirs(os.path.dirname(out_path))
        # np.save(out_path, shifts_and_rms[:, :, 2])
        # short_path = input_path[len(output_dir) :]
        # print(f"> $OUTPUT{short_path}")
        raise NotImplementedError("This method is not implemented yet.")

    def _save_block_alignments(self, shifts_and_rms, output_dir, input_path):
        raise NotImplementedError("This method is not implemented yet.")


class CompareBlockGlobal(Module):
    def __init__(self, registration_params: RegistrationParams):
        super().__init__(
            input_type=DataType.MATRIX_3D,
            output_type=DataType.SHIFT_TUPLE,
            reference_type=DataType.IMAGE_2D,
            supplementary_type=DataType.IMAGE_2D,
        )
        self.dirname = "register_global"
        self.reference_data = None
        self.ref_fiducial = registration_params.referenceFiducial
        self.background_sigma = registration_params.background_sigma
        self.align_by_block = registration_params.alignByBlock
        self.block_size = registration_params.blockSize
        self.tolerance = registration_params.tolerance

    def run(self, shifts_and_rms, img_to_align):
        if shifts_and_rms is None:
            print("> No need to align reference image.")
            return None
        preprocessed_img = remove_inhomogeneous_background(
            img_to_align, self.background_sigma
        )
        ref_img = np.float32(self.reference_data)
        raw_img = np.float32(match_histograms(np.float32(preprocessed_img), ref_img))
        # mask = get_rms_mask(shifts_and_rms[:, :, 2], self.tolerance)
        # masked_block_shifts = np.where(mask[..., None], shifts_and_rms[:, :, 2], np.nan)
        # relative_shifts = compute_relative_shifts(masked_block_shifts)
        return compare_to_global(ref_img, raw_img, self.tolerance, shifts_and_rms)

    def load_data(self, input_path, in_dir_length):
        if self.ref_fiducial in os.path.basename(input_path):
            return None
        print(f"[Load] {self.input_type.value}")
        short_path = input_path[in_dir_length:]
        print(f"> $INPUT{short_path}")
        return np.load(input_path)

    def save_data(self, data, output_dir, input_path):
        if data is None:
            return
        print("Saving shift tuple.")
        self._save_shift_tuple(data, output_dir, input_path)
        raw_img = np.load(input_path)
        preprocessed_img = remove_inhomogeneous_background(
            raw_img, self.background_sigma
        )
        ref_img = np.float32(self.reference_data)
        raw_img = np.float32(match_histograms(np.float32(preprocessed_img), ref_img))
        shifted_img = shift_image(raw_img, data)
        self._save_registered(shifted_img, output_dir, input_path)
        self._save_overlay_corrected(ref_img, shifted_img, output_dir, input_path)
        self._save_reference_difference(raw_img, shifted_img, output_dir, input_path)

    def load_reference_data(self, paths: List[str]):
        good_path = None
        for path in paths:
            if self.ref_fiducial in os.path.basename(path):
                good_path = path
                break
        if good_path[-3:] == "npy":
            ref_img = np.load(good_path)
            self.reference_data = remove_inhomogeneous_background(
                ref_img, self.background_sigma
            )
        else:
            raise NotImplementedError("Reference data must be a 2D numpy file.")
        
    def load_supplementary_data(self, data_type, cycle):
        if cycle == self.ref_fiducial:
            return []
        raise NotImplementedError("This method is not implemented yet.")

    def _save_shift_tuple(self, shifts, output_dir, input_path):
        out_path = os.path.join(output_dir, self.dirname, "data", "shifts.json")
        cycle = DataManager.get_cycle_from_path(input_path)
        roi = get_roi_number_from_image_path(input_path)
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        if not os.path.exists(out_path):
            existing_dict = {f"ROI:{roi}": {}}
        else:
            existing_dict = load_json(out_path)
        existing_dict[f"ROI:{roi}"][cycle] = shifts.tolist()
        with open(out_path, "w") as file:
            json.dump(existing_dict, file, ensure_ascii=False, sort_keys=True, indent=4)

    def _save_registered(self, shifted_img, output_dir, input_path):
        base = os.path.basename(input_path).split(".")[0]
        npy_filename = base + "_2d_registered.npy"
        npy_path = os.path.join(output_dir, self.dirname, "data", npy_filename)
        save_npy(shifted_img, npy_path, len(output_dir))

    def _save_overlay_corrected(self, ref_img, shifted_img, output_dir, input_path):
        base = os.path.basename(input_path).split(".")[0]
        png_filename = base + "_overlay_corrected.png"
        png_path = os.path.join(output_dir, self.dirname, png_filename)
        sz = ref_img.shape
        img_1, img_2 = (
            ref_img / ref_img.max(),
            shifted_img / shifted_img.max(),
        )
        img_1, _, _, _, _ = image_adjust(
            img_1, lower_threshold=0.5, higher_threshold=0.9999
        )
        img_2, _, _, _, _ = image_adjust(
            img_2, lower_threshold=0.5, higher_threshold=0.9999
        )
        fig, ax1 = plt.subplots()
        fig.set_size_inches((30, 30))
        null_image = np.zeros(sz)
        rgb = np.dstack([img_1, img_2, null_image])
        ax1.imshow(rgb)
        ax1.axis("off")
        fig.savefig(png_path)
        plt.close(fig)

    def _save_reference_difference(self, raw_img, shifted_img, output_dir, input_path):
        raise NotImplementedError("This method is not implemented yet.")


def remove_inhomogeneous_background(img, background_sigma):
    # Normalises images
    norm_img = img / img.max()
    # removes inhomogeneous background
    sigma_clip = SigmaClip(sigma=background_sigma)
    bkg_estimator = MedianBackground()
    bkg = Background2D(
        norm_img,
        (64, 64),
        filter_size=(3, 3),
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator,
    )
    return norm_img - bkg.background


def align_images_by_blocks(ref_img, img_to_align, block_size, tolerance=0.1):
    shifts_and_rms = compute_shifts_and_rms_by_block(ref_img, img_to_align, block_size)
    mean_shifts = compare_to_global(ref_img, img_to_align, tolerance, shifts_and_rms)
    return np.array(mean_shifts), shifts_and_rms


def compute_shifts_and_rms_by_block(ref_img, raw_img, block_size):
    block_size_tuple = (block_size, block_size)
    ref_blocks = view_as_blocks(ref_img, block_size_tuple)
    raw_blocks = view_as_blocks(raw_img, block_size_tuple)
    i_blocks = ref_blocks.shape[0]
    j_blocks = ref_blocks.shape[1]
    blocks = np.zeros((i_blocks, j_blocks, 3))
    for i in trange(i_blocks):
        for j in range(j_blocks):
            shift, _, _ = phase_cross_correlation(
                ref_blocks[i, j], raw_blocks[i, j], upsample_factor=100
            )
            blocks[i, j, 0], blocks[i, j, 1] = shift
            tmp_aligned_img = shift_image(raw_img, shift)
            blocks[i, j, 2] = np.sum(np.sum(np.abs(ref_img - tmp_aligned_img), axis=1))
    return blocks


def compute_relative_shifts(masked_block_shifts):
    i_blocks = masked_block_shifts.shape[0]
    j_blocks = masked_block_shifts.shape[1]
    shift_image_norm = np.zeros((i_blocks, j_blocks))
    for i in range(i_blocks):
        for j in range(j_blocks):
            if np.isnan(masked_block_shifts[i, j][0]):
                shift_image_norm[i, j] = np.nan
            else:
                shift_image_norm[i, j] = LA.norm(masked_block_shifts[i, j])
    mean_shift_norm = np.mean(shift_image_norm[~np.isnan(shift_image_norm)])
    relative_shifts = np.abs(shift_image_norm - mean_shift_norm)
    mean_relative_shifts = np.mean(relative_shifts[~np.isnan(relative_shifts)])
    print(f"*** Mean relative shifts: {mean_relative_shifts:.2f} px")
    return relative_shifts


def get_rms_mask(rms_image, tolerance):
    threshold = (1 + tolerance) * np.min(rms_image)
    return rms_image < threshold


def compare_to_global(ref_img, img_to_align, tolerance, shifts_and_rms):

    # [calculates global shift, if it is better than the polled shift, or
    # if we do not have enough pollsters to fall back to then it does a global cross correlation!]
    mean_shifts_global, _, _ = phase_cross_correlation(
        ref_img, img_to_align, upsample_factor=100
    )
    img_2_aligned_global = shift_image(img_to_align, mean_shifts_global)
    mean_error_global = np.sum(np.sum(np.abs(ref_img - img_2_aligned_global), axis=1))
    mean_error_raw = np.sum(np.sum(np.abs(ref_img - img_to_align), axis=1))

    mask = get_rms_mask(shifts_and_rms[:, :, 2], tolerance)
    mean_error_block = np.mean(shifts_and_rms[:, :, 2][mask])
    print(f"> Block alignment error: {mean_error_block}")
    print(f"> Global alignment error: {mean_error_global}")
    print(f"> Global raw error: {mean_error_raw}")
    # [Averages shifts and errors from regions within the tolerated blocks]
    std_shifts = [np.std(shifts_and_rms[mask, 0]), np.std(shifts_and_rms[mask, 1])]
    mean_shifts = [np.mean(shifts_and_rms[mask, 0]), np.mean(shifts_and_rms[mask, 1])]

    min_number_pollsters = 4
    shift_error_tolerance = 5
    if (
        np.sum(mask) < min_number_pollsters
        or mean_error_global < mean_error_block
        or np.max(std_shifts) > shift_error_tolerance
    ):
        mean_shifts = mean_shifts_global
        print("Falling back to global registration")

    print(
        f"*** Global XY shifts: {mean_shifts_global[0]:.2f} px | {mean_shifts_global[1]:.2f} px"
    )

    print(
        f"*** Mean polled XY shifts: {mean_shifts[0]:.2f}({std_shifts[0]:.2f}) px | {mean_shifts[1]:.2f}({std_shifts[1]:.2f}) px"
    )
    return mean_shifts


def image_adjust(image, lower_threshold=0.3, higher_threshold=0.9999):
    """
    Adjust intensity levels:
        - rescales exposures
        - gets histogram of pixel intensities to define cutoffs
        - applies thresholds

    Parameters
    ----------
    image : numpy array
        input 3D image.
    lower_threshold : float, optional
        lower threshold for adjusting image levels. The default is 0.3.
    higher_threshold : float, optional
        higher threshold for adjusting image levels.. The default is 0.9999.

    Returns
    -------
    image1 : numpy array
        adjusted 3D image.
    hist1_before : numpy array
        histogram of pixel intensities before adjusting levels.
    hist1 : numpy array
        histogram of pixel intensities after adjusting levels.
    lower_cutoff : float
        lower cutoff used for thresholding.
    higher_cutoff : float
        higher cutoff used for thresholding.

    """
    # print_log("> Rescaling grey levels...")

    # rescales image to [0,1]
    image1 = exposure.rescale_intensity(image, out_range=(0, 1))

    # calculates histogram of intensities
    hist1_before = exposure.histogram(image1)

    hist_sum = np.zeros(len(hist1_before[0]))
    for i in range(len(hist1_before[0]) - 1):
        hist_sum[i + 1] = hist_sum[i] + hist1_before[0][i]

    sum_normalized = hist_sum / hist_sum.max()
    lower_cutoff = np.where(sum_normalized > lower_threshold)[0][0] / 255
    higher_cutoff = np.where(sum_normalized > higher_threshold)[0][0] / 255

    # adjusts image intensities from (lower_threshold,higher_threshold) --> [0,1]
    image1 = exposure.rescale_intensity(
        image1, in_range=(lower_cutoff, higher_cutoff), out_range=(0, 1)
    )

    # calculates histogram of intensities of adjusted image
    hist1 = exposure.histogram(image1)

    return image1, hist1_before, hist1, lower_cutoff, higher_cutoff
