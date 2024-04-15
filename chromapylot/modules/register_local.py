#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

from chromapylot.modules.module import Module
from chromapylot.core.data_manager import DataManager
from chromapylot.parameters.registration_params import RegistrationParams
from chromapylot.core.core_types import DataType


class RegisterLocal(Module):
    def __init__(
        self, data_manager: DataManager, registration_params: RegistrationParams
    ):
        super().__init__(
            data_manager=data_manager,
            input_type=DataType.IMAGE_3D_SHIFTED,
            output_type=DataType.REGISTRATION_TABLE,
            reference_type=DataType.IMAGE_3D,
            supplementary_type=None,
        )
        self.block_size_xy = registration_params.blockSizeXY

    def load_data(self, input_path):
        return self.data_m.load_image_3d(input_path)

    def load_reference_data(self, paths: List[str]):
        good_path = None
        for path in paths:
            if self.ref_fiducial in os.path.basename(path):
                good_path = path
                break
        if good_path and good_path[-3:] == "tif":
            self.reference_data = self.data_m.load_image_3d(good_path)
        else:
            raise NotImplementedError("Reference data must be a 3D tif file")

    def run(self, data):
        # - break in blocks
        num_planes = data[0].shape[0]
        block_size = (num_planes, self.block_size_xy, self.block_size_xy)

        print("$ Breaking images into blocks")
        ref_blocks = view_as_blocks(
            self.reference_data, block_shape=block_size
        ).squeeze()
        img_blocks = view_as_blocks(data, block_shape=block_size).squeeze()

        # - loop thru blocks and calculates block shift in xyz:
        shift_matrices = [np.zeros(ref_blocks.shape[:2]) for _ in range(3)]

        for i in trange(block_ref.shape[0]):
            for j in range(block_ref.shape[1]):
                # - cross correlate in 3D to find 3D shift
                shifts_xyz, _, _ = phase_cross_correlation(
                    block_ref[i, j], block_target[i, j], upsample_factor=upsample_factor
                )
                for matrix, _shift in zip(shift_matrices, shifts_xyz):
                    matrix[i, j] = _shift

        return shift_matrices

    def save_data(self, data, input_path):
        raise NotImplementedError
        if data is None:
            return
        self._save_shift_tuple(data, input_path)
        raw_img = np.load(input_path)
        shifted_3d_img = shift_image(raw_img, data)
        self._save_3d_alignments(block_ref, block_shifted)  # fig3 outputs
        self._save_bkg_substracted()  # fig1 plot_4_images(...)
        self._save_mse_blocks()  # fig5 plot_3d_shift_matrices(...)
        self._save_shift_matrices(data)  # fig2 plot_3d_shift_matrices(...)
