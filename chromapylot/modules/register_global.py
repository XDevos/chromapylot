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


class RegisterGlobalModule(Module):
    def __init__(self, registration_params: RegistrationParams):
        super().__init__(
            input_type=DataType.IMAGE_2D,
            output_type=DataType.SHIFT_TUPLE,
            reference_type=DataType.IMAGE_2D,
        )
        self.reference_data = None
        self.ref_fiducial = registration_params.referenceFiducial
        self.background_sigma = registration_params.background_sigma
        self.align_by_block = registration_params.alignByBlock
        self.block_size = registration_params.blockSize
        self.tolerance = registration_params.tolerance
        self.dirname = "register_global"

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
                block_shifts,
                rms_image,
            ) = align_images_by_blocks(
                ref_img,
                raw_img,
                self.block_size,
                tolerance=self.tolerance,
            )
            return {
                "shifts": shifts,
                "block_shifts": block_shifts,
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
        ref_img = np.load(good_path)
        self.reference_data = remove_inhomogeneous_background(
            ref_img, self.background_sigma
        )

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
    block_size_tuple = (block_size, block_size)
    block_1 = view_as_blocks(ref_img, block_size_tuple)
    block_2 = view_as_blocks(img_to_align, block_size_tuple)
    shift_image_norm = np.zeros((block_1.shape[0], block_1.shape[1]))
    shifted_image = np.zeros((block_1.shape[0], block_1.shape[1], 2))
    rms_image = np.zeros((block_1.shape[0], block_1.shape[1]))

    for i in trange(block_1.shape[0]):
        for j in range(block_1.shape[1]):
            # using Scimage registration functions
            shift, _, _ = phase_cross_correlation(
                block_1[i, j], block_2[i, j], upsample_factor=100
            )
            shift_image_norm[i, j] = LA.norm(shift)
            shifted_image[i, j, 0], shifted_image[i, j, 1] = shift[0], shift[1]
            img_2_aligned = shift_image(img_to_align, shift)
            rms_image[i, j] = np.sum(np.sum(np.abs(ref_img - img_2_aligned), axis=1))

    threshold = (1 + tolerance) * np.min(rms_image)
    mask = rms_image < threshold

    # [Averages shifts and errors from regions within the tolerated blocks]
    mean_shifts = [np.mean(shifted_image[mask, 0]), np.mean(shifted_image[mask, 1])]
    std_shifts = [np.std(shifted_image[mask, 0]), np.std(shifted_image[mask, 1])]
    mean_shift_norm = np.mean(shift_image_norm[mask])
    mean_error = np.mean(rms_image[mask])
    relative_shifts = np.abs(shift_image_norm - mean_shift_norm)

    # [calculates global shift, if it is better than the polled shift, or
    # if we do not have enough pollsters to fall back to then it does a global cross correlation!]
    mean_shifts_global, _, _ = phase_cross_correlation(
        ref_img, img_to_align, upsample_factor=100
    )
    img_2_aligned_global = shift_image(img_to_align, shift)
    mean_error_global = np.sum(np.sum(np.abs(ref_img - img_2_aligned_global), axis=1))

    print(
        f"Block alignment error: {mean_error}, global alignment error: {mean_error_global}"
    )

    min_number_pollsters = 4
    shift_error_tolerance = 5
    if (
        np.sum(mask) < min_number_pollsters
        or mean_error_global < mean_error
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

    return np.array(mean_shifts), relative_shifts, rms_image
