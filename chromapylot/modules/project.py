#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from core_types import DataType
from module import Module
from parameters import ProjectionParams
from skimage.util.shape import view_as_blocks
from tqdm import trange
from scipy.stats import sigmaclip
import os
from skimage import filters
from skimage import io
import scipy.optimize as spo
from data_manager import save_npy


class ProjectModule(Module):
    def __init__(self, projection_params: ProjectionParams):
        super().__init__(input_type=DataType.IMAGE_3D, output_type=DataType.IMAGE_2D)
        self.mode = projection_params.mode
        self.dirname = "project"
        self.block_size = projection_params.block_size
        self.zwindows = projection_params.zwindows

    def load_data(self, input_path):
        return io.imread(input_path).squeeze()

    def save_data(self, data, output_dir, input_path):
        base = os.path.basename(input_path).split(".")[0]
        out_name = base + "_2d.npy"
        path_name = os.path.join(output_dir, self.dirname, "data", out_name)
        save_npy(data, path_name)

    def run(self, array_3d):
        if self.mode == "laplacian":
            img_projected, (
                focal_plane_matrix,
                focus_plane,
            ) = self._projection_laplacian(array_3d)
            return img_projected

    def _projection_laplacian(self, img):
        focal_plane_matrix, z_range, block = reinterpolate_focal_plane(
            img, block_size_xy=self.block_size, window=self.zwindows
        )
        # reassembles image
        output = reassemble_images(focal_plane_matrix, block, window=self.zwindows)

        return output, (focal_plane_matrix, z_range[0])


# =============================================================================
# FOCAL PLANE INTERPOLATION
# =============================================================================


def reinterpolate_focal_plane(data, block_size_xy=256, window=10):
    """
    Reinterpolates the focal plane of a 3D image by breking it into blocks
    - Calculates the focal_plane and fwhm matrices by block
    - removes outliers
    - calculates the focal plane for each block using sigmaClip statistics
    - returns a tuple with focal plane and the range to use

    Parameters
    ----------
    data : numpy array
        input 3D image.
    block_size_xy : int
        size of blocks in XY, typically 256.
    window : int, optional
        number of planes before and after the focal plane to construct the z_range. The default is 0.

    Returns
    -------
    focal_plane_matrix : numpy array
        focal plane matrix.
    z_range : tuple
        focus_plane, z_range.
    block : numpy array
        block representation of 3D image.

    """

    # breaks into subplanes, iterates over them and calculates the focal_plane in each subplane.
    focal_plane_matrix, block = calculate_focus_per_block(
        data, block_size_xy=block_size_xy
    )
    focal_planes_to_process = focal_plane_matrix[~np.isnan(focal_plane_matrix)]

    focal_plane, _, _ = sigmaclip(focal_planes_to_process, high=3, low=3)
    focus_plane = np.mean(focal_plane)
    if np.isnan(focus_plane):
        # focus_plane detection failed. Using full stack.
        focus_plane = data.shape[0] // 2
        z_range = focus_plane, range(0, data.shape[0])
    else:
        focus_plane = np.mean(focal_plane).astype("int64")
        zmin = np.max([focus_plane - window, 0])
        zmax = np.min([focus_plane + window, data.shape[0]])
        z_range = focus_plane, range(zmin, zmax)

    return focal_plane_matrix, z_range, block


def calculate_focus_per_block(data, block_size_xy=128):
    """
    Calculates the most likely focal plane of an image by breaking into blocks and calculating
    the focal plane in each block

    - breaks image into blocks
    - returns the focal plane + fwhm for each block

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    block_size_xy : TYPE, optional
        DESCRIPTION. The default is 512.

    Returns
    -------
    focal_plane_matrix: np array
        matrix containing the maximum of the laplacian variance per block
    block: np array
        3D block reconstruction of matrix
    """

    n_z_planes = data.shape[0]
    block_size = (n_z_planes, block_size_xy, block_size_xy)
    block = view_as_blocks(data, block_size)
    # block.shape ~= (n_z_blocks, n_x_blocks, n_y_blocks, n_z_planes, 128, 128)
    # But n_z_blocks = 1 due to we give the full z stack to the function
    block = block.squeeze()  # So we remove the first dimension
    focal_plane_matrix = np.zeros(block.shape[0:2])  # (n_x_blocks, n_y_blocks)

    for i in trange(block.shape[0]):
        for j in range(block.shape[1]):
            focal_plane_matrix[i, j] = find_focal_plane(block[i, j])

    return focal_plane_matrix, block


def find_focal_plane(data, threshold_fwhm=20):
    """
    This function will find the focal plane of a 3D image
    - calculates the laplacian variance of the image for each z plane
    - fits 1D gaussian profile on the laplacian variance
    - to get the maximum (focal plane) and the full width at half maximum
    - it returns nan if the fwhm > threshold_fwhm (means fit did not converge)
    - threshold_fwhm should represend the width of the laplacian variance curve
    - which is often 5-10 planes depending on sample.

    Parameters
    ----------
    data : numpy array
        input 3D image ZYX.
    threshold_fwhm : float, optional
        threshold fwhm used to remove outliers. The default is 20.

    Returns
    -------
    focal_plane : float
        focal plane: max of fitted z-profile.

    """
    # finds focal plane
    raw_images = [data[i, :, :] for i in range(data.shape[0])]
    laplacian_variance = [np.var(filters.laplace(img)) for img in raw_images]
    laplacian_variance = laplacian_variance / max(laplacian_variance)
    x_coord = range(len(laplacian_variance))
    fit_result = fit_1d_gaussian_scipy(
        x_coord,
        laplacian_variance,
        title="laplacian variance z-profile",
    )
    if len(fit_result) > 0 and fit_result["gauss1d.fwhm"] <= threshold_fwhm:
        return fit_result["gauss1d.pos"]
    return np.nan


# Gaussian function
# @jit(nopython=True)
def gaussian(x, a=1, mean=0, std=0.5):
    return (
        a
        * (1 / (std * (np.sqrt(2 * np.pi))))
        * (np.exp(-((x - mean) ** 2) / ((2 * std) ** 2)))
    )


def fit_1d_gaussian_scipy(x, y, title=""):
    """
    Fits a function using a 1D Gaussian and returns parameters if successfull.
    Otherwise will return an empty dict
    Uses scipy spo package

    Parameters
    ----------
    x : numpy 1D array
        x data.
    y : numpy 1D array
        y data.
    title : str, optional
        figure title. The default is ''.

    Returns
    -------
    {}
        dictionary with fitting parameters.

    """
    fit_result = {}

    try:
        fitgauss = spo.curve_fit(gaussian, x, y)
        fit_result["gauss1d.pos"] = fitgauss[0][1]
        fit_result["gauss1d.ampl"] = fitgauss[0][0]
        fit_result["gauss1d.fwhm"] = 2.355 * fitgauss[0][2]
    except RuntimeError:
        return {}
    except ValueError:
        fit_result["gauss1d.pos"] = np.mean(x)
        fit_result["gauss1d.ampl"] = 0.0
        fit_result["gauss1d.fwhm"] = 0.0
        # Returned middle plane
        return fit_result

    return fit_result


def reassemble_images(focal_plane_matrix, block, window=0):
    """
    Makes 2D image from 3D stack by reassembling sub-blocks
    For each sub-block we know the optimal focal plane, which is
    selected for the assembly of the while image

    Parameters
    ----------
    focal_plane_matrix : numpy 2D array
        matrix containing the focal plane selected for each block.
    block : numpy matrix
        original 3D image sorted by blocks.

    Returns
    -------
    output : numpy 2D array
        output 2D projection

    """

    # Fix ValueError: cannot convert float NaN to integer
    focal_plane_matrix = np.nan_to_num(focal_plane_matrix)
    # gets image size from block image
    number_blocks = block.shape[0]
    block_size_xy = block.shape[3]
    im_size = number_blocks * block_size_xy

    # gets ranges for slicing
    slice_coordinates = [
        range(x * block_size_xy, (x + 1) * block_size_xy) for x in range(number_blocks)
    ]

    # creates output image
    output = np.zeros((im_size, im_size))

    # gets more common plane
    focal_planes = []
    for i, i_slice in enumerate(slice_coordinates):
        for j, j_slice in enumerate(slice_coordinates):
            focal_planes.append(int(focal_plane_matrix[i, j]))
    most_common_focal_plane = max(set(focal_planes), key=focal_planes.count)

    # reassembles image
    if window == 0:
        # takes one plane block
        for i, i_slice in enumerate(slice_coordinates):
            for j, j_slice in enumerate(slice_coordinates):
                focus = int(focal_plane_matrix[i, j])
                if np.abs(focus - most_common_focal_plane) > 1:
                    focus = int(most_common_focal_plane)
                output[i_slice[0] : i_slice[-1] + 1, j_slice[0] : j_slice[-1] + 1] = (
                    block[i, j][focus, :, :]
                )
    else:
        # takes neighboring planes by projecting
        for i, i_slice in enumerate(slice_coordinates):
            for j, j_slice in enumerate(slice_coordinates):
                focus = int(focal_plane_matrix[i, j])
                if np.abs(focus - most_common_focal_plane) > 1:
                    focus = int(most_common_focal_plane)
                zmin = np.max((0, focus - round(window / 2)))
                zmax = np.min((block[i, j].shape[0], focus + round(window / 2)))
                block_img = block[i, j][:, :, :]
                output[i_slice[0] : i_slice[-1] + 1, j_slice[0] : j_slice[-1] + 1] = (
                    block_img[zmin:zmax].max(axis=0)
                )

    return output
