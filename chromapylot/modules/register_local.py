#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import List, Union
import matplotlib

from tqdm import tqdm, trange
import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.util.shape import view_as_blocks
from scipy.ndimage import shift as shift_image
from skimage import exposure
from astropy.table import Table, vstack
from dask.distributed import Lock
import matplotlib.pylab as plt
from matplotlib import ticker
from skimage.metrics import mean_squared_error, normalized_root_mse

from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground
from chromapylot.modules.module import Module
from chromapylot.core.data_manager import (
    DataManager,
    get_roi_number_from_image_path,
    create_png_path,
)
from chromapylot.parameters.registration_params import RegistrationParams
from chromapylot.parameters.segmentation_params import SegmentationParams
from chromapylot.core.core_types import DataType
from chromapylot.modules.register_global import image_adjust

font = {"weight": "normal", "size": 22}
matplotlib.rc("font", **font)


class Preprocess3D(Module):
    def __init__(
        self,
        data_manager: DataManager,
        registration_params: RegistrationParams,
        segmentation_params: SegmentationParams,
    ):
        super().__init__(
            data_manager=data_manager,
            input_type=DataType.IMAGE_3D,
            output_type=DataType.IMAGE_3D,
            reference_type=None,
            supplementary_type=DataType.REDUCE_TUPLE,
        )
        self._3D_lower_threshold = registration_params._3D_lower_threshold
        self._3D_higher_threshold = registration_params._3D_higher_threshold
        self.reduce_planes = segmentation_params.reducePlanes
        self.method_3d = segmentation_params._3Dmethod
        if self.reduce_planes and self.method_3d == "thresholding":
            self._3D_lower_threshold = segmentation_params._3D_lower_threshold
            self._3D_higher_threshold = segmentation_params._3D_higher_threshold

    def load_data(self, input_path):
        return self.data_m.load_image_3d(input_path)

    def run(self, data, supplementary_type=None):
        if self.reduce_planes:
            zmin, zmax = supplementary_type
            data = data[zmin:zmax, :, :]
        if self.method_3d != "stardist":
            data = exposure.rescale_intensity(data, out_range=(0, 1))
            data = remove_inhomogeneous_background_3d(data)
            data = image_adjust(
                data, self._3D_lower_threshold, self._3D_higher_threshold
            )
        return data

    def save_data(self, data, input_path, input_data, supplementary_data):
        pass

    def load_supplementary_data(self, input_path, cycle):
        if not self.reduce_planes:
            return None
        else:
            raise NotImplementedError("Reduce planes loading not implemented yet")


class RegisterLocal(Module):
    def __init__(
        self, data_manager: DataManager, registration_params: RegistrationParams
    ):
        super().__init__(
            data_manager=data_manager,
            input_type=[DataType.IMAGE_3D_SHIFTED, DataType.IMAGE_3D],
            output_type=DataType.REGISTRATION_TABLE,
            reference_type=DataType.IMAGE_3D,
            supplementary_type=None,
        )
        self.dirname = "register_local"
        self.block_size_xy = registration_params.blockSizeXY
        self.upsample_factor = registration_params.upsample_factor
        self.ref_fiducial = registration_params.referenceFiducial
        self._3D_lower_threshold = registration_params._3D_lower_threshold
        self._3D_higher_threshold = registration_params._3D_higher_threshold

    def load_data(self, input_path):
        return self.data_m.load_image_3d(input_path)

    def load_reference_data(self, paths: List[str]):
        good_path = None
        for path in paths:
            if self.ref_fiducial in os.path.basename(path):
                good_path = path
                break
        if good_path and good_path[-3:] == "tif":
            ref_3d = self.data_m.load_image_3d(good_path)
            # TODO: it's a (may be long) tempo fix to apply SkipModule at the ref 3d fiducial
            skipped_ref = ref_3d[::2, :, :]
            preproc_ref = self._preprocess_data(skipped_ref)
            self.reference_data = preproc_ref
        else:
            raise NotImplementedError("Reference data must be a 3D tif file")

    def run(self, data):
        # data = self._preprocess_data(data)
        # - break in blocks
        num_planes = data.shape[0]
        block_size = (num_planes, self.block_size_xy, self.block_size_xy)

        print("$ Breaking images into blocks")
        ref_blocks = view_as_blocks(
            self.reference_data, block_shape=block_size
        ).squeeze()
        img_blocks = view_as_blocks(data, block_shape=block_size).squeeze()

        # - loop thru blocks and calculates block shift in xyz:
        shift_matrices = [np.zeros(ref_blocks.shape[:2]) for _ in range(3)]

        for i in trange(ref_blocks.shape[0]):
            for j in range(ref_blocks.shape[1]):
                # - cross correlate in 3D to find 3D shift
                shifts_xyz, _, _ = phase_cross_correlation(
                    ref_blocks[i, j],
                    img_blocks[i, j],
                    upsample_factor=self.upsample_factor,
                )
                for matrix, _shift in zip(shift_matrices, shifts_xyz):
                    matrix[i, j] = _shift
        registration_table = self.create_registration_table(shift_matrices)
        return registration_table

    def save_data(
        self, registration_table, input_path, shifted_3d_img, supplementary_data
    ):
        if registration_table is None:
            return
        shift_matrices = self.table_to_shift_matrices(registration_table)
        shifted_blocks_by_axis, mse_matrices, nrmse_matrices = (
            self.shift_and_project_blocks(shift_matrices, shifted_3d_img)
        )
        self._save_registration_table(registration_table, input_path, nrmse_matrices)
        png_path = create_png_path(
            input_path, self.data_m.output_folder, self.dirname, ".tif_3Dalignments"
        )
        self._save_3d_alignments(shifted_blocks_by_axis, png_path)

        png_path = create_png_path(
            input_path, self.data_m.output_folder, self.dirname, ".tif_MSEblocks"
        )
        self._save_mse_blocks(mse_matrices, png_path)
        png_path = create_png_path(
            input_path, self.data_m.output_folder, self.dirname, ".tif_shiftMatrices"
        )
        self._save_shift_matrices(png_path, shift_matrices)

    def shift_and_project_blocks(self, shift_matrices, shifted_3d_img):
        num_planes = shifted_3d_img.shape[0]
        block_size = (num_planes, self.block_size_xy, self.block_size_xy)
        print("$ Breaking images into blocks")
        ref_blocks = view_as_blocks(
            self.reference_data, block_shape=block_size
        ).squeeze()
        img_blocks = view_as_blocks(shifted_3d_img, block_shape=block_size).squeeze()
        # combines blocks into a single matrix for display instead of plotting a matrix of subplots each with a block
        blocks_by_axis = []
        mse_blocks = []
        nrmse_matrices = []
        for axis in range(3):
            output = combine_blocks_image_by_reprojection(
                ref_blocks, img_blocks, shift_matrices=shift_matrices, axis1=axis
            )
            blocks_by_axis.append(output[0])
            mse_blocks.append(output[1])
            nrmse_matrices.append(output[2])
        return blocks_by_axis, mse_blocks, nrmse_matrices

    def _save_registration_table(self, data, input_path, nrmse_matrices):
        data = self.__update_registration_table(data, input_path, nrmse_matrices)
        out_path = os.path.join(
            self.data_m.output_folder, self.dirname, "data", "shifts_block3D.dat"
        )
        try:
            with Lock(out_path):
                self.__save_registration_table(data, out_path)
        except (RuntimeError, ValueError):
            self.__save_registration_table(data, out_path)

    def __update_registration_table(self, data, input_path, nrmse_matrices):
        filename = os.path.basename(input_path)
        # update with aligned file
        data["aligned file"] = [filename] * len(data)
        # update with ROI number
        data["ROI #"] = get_roi_number_from_image_path(input_path)
        # update with cycle number
        cycle = self.data_m.get_cycle_from_path(input_path)
        data["label"] = [cycle] * len(data)
        # update with NRMSE matrices
        num_blocks = nrmse_matrices[0].shape[0]
        for i in range(num_blocks):
            for j in range(num_blocks):
                data["quality_xy"][i * num_blocks + j] = nrmse_matrices[0][i, j]
                data["quality_zy"][i * num_blocks + j] = nrmse_matrices[1][i, j]
                data["quality_zx"][i * num_blocks + j] = nrmse_matrices[2][i, j]
        return data

    def __save_registration_table(self, data, out_path):
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        elif os.path.exists(out_path):
            existing_table = Table.read(out_path, format="ascii.ecsv")
            data = vstack([existing_table, data])
        data.write(out_path, format="ascii.ecsv", overwrite=True)

    def _save_3d_alignments(self, shifted_blocks_by_axis, png_path):
        fig = plt.figure(constrained_layout=False)
        fig.set_size_inches((20 * 2, 20))
        grid_spec = fig.add_gridspec(2, 2)
        ax = [
            fig.add_subplot(grid_spec[:, 0]),
            fig.add_subplot(grid_spec[0, 1]),
            fig.add_subplot(grid_spec[1, 1]),
        ]
        titles = ["Z-projection", "X-projection", "Y-projection"]
        for axis, output, i in zip(ax, shifted_blocks_by_axis, range(3)):
            axis.imshow(output)
            axis.set_title(titles[i])
        fig.tight_layout()
        fig.savefig(png_path)
        plt.close(fig)

    def _save_mse_blocks(self, mse_matrices, png_path):
        self._save_value_blocks_by_axis(
            mse_matrices,
            png_path,
            subtitle_str="mean square error block matrices",
            fontsize=6,
            valfmt="{x:.2f}",
        )

    def _save_shift_matrices(self, path, shift_matrices):
        self._save_value_blocks_by_axis(
            shift_matrices,
            path,
            subtitle_str=None,
            fontsize=8,
            valfmt="{x:.1f}",
        )

    def _save_value_blocks_by_axis(
        self,
        value_blocks,
        path,
        subtitle_str=None,
        fontsize=6,
        valfmt="{x:.2f}",
    ):

        cbar_kw = {"fraction": 0.046, "pad": 0.04}
        fig, axes = plt.subplots(1, len(value_blocks))
        fig.set_size_inches((len(value_blocks) * 10, 10))
        ax = axes.ravel()
        titles = ["z shift matrix", "x shift matrix", "y shift matrix"]

        for axis, title, matrix in zip(ax, titles, value_blocks):
            row = [str(x) for x in range(matrix.shape[0])]
            im, _ = heatmap(
                matrix,
                row,
                row,
                ax=axis,
                cmap="YlGn",
                cbarlabel=title,
                fontsize=fontsize,
                cbar_kw=cbar_kw,
            )
            annotate_heatmap(
                im,
                valfmt=valfmt,
                size=fontsize,
                threshold=None,
                textcolors=("black", "white"),
            )
            axis.set_title(title)
        if subtitle_str:
            fig.suptitle(subtitle_str)
        fig.savefig(path)
        plt.close(fig)

    def _preprocess_data(self, data):
        data = exposure.rescale_intensity(data, out_range=(0, 1))
        img = remove_inhomogeneous_background_3d(data)
        adjust_img = image_adjust(
            img, self._3D_lower_threshold, self._3D_higher_threshold
        )
        return adjust_img

    def create_registration_table(self, shift_matrices):
        alignment_results_table = create_output_table()
        num_blocks = shift_matrices[0].shape[0]
        for i in range(num_blocks):
            for j in range(num_blocks):
                table_entry = [
                    self.data_m.get_3d_ref_filename(),
                    "",
                    int(self.block_size_xy),
                    0,
                    "",
                    i,
                    j,
                    shift_matrices[0][i, j],
                    shift_matrices[1][i, j],
                    shift_matrices[2][i, j],
                    0.0,
                    0.0,
                    0.0,
                ]
                alignment_results_table.add_row(table_entry)
        return alignment_results_table

    def table_to_shift_matrices(self, table):
        shift_matrices = [
            np.zeros((table["block_i"].max() + 1, table["block_j"].max() + 1))
            for _ in range(3)
        ]
        for i, row in enumerate(table):
            for j, axis in enumerate(["shift_z", "shift_x", "shift_y"]):
                shift_matrices[j][row["block_i"], row["block_j"]] = row[axis]
        return shift_matrices


def create_output_table():
    return Table(
        names=(
            "reference file",
            "aligned file",
            "blockXY",
            "ROI #",
            "label",
            "block_i",
            "block_j",
            "shift_z",
            "shift_x",
            "shift_y",
            "quality_xy",
            "quality_zy",
            "quality_zx",
        ),
        dtype=(
            "S2",
            "S2",
            "int",
            "int",
            "S2",
            "int",
            "int",
            "f4",
            "f4",
            "f4",
            "f4",
            "f4",
            "f4",
        ),
    )


def remove_inhomogeneous_background_3d(image_3d, box_size=(32, 32), filter_size=(3, 3)):
    """
    Wrapper to remove inhomogeneous background in a 3D image by recursively calling _remove_inhomogeneous_background_2d():
        - addresses output
        - iterates over planes and calls _remove_inhomogeneous_background_2d in each plane
        - reassembles results into a 3D image

    Parameters
    ----------
    image_3d : numpy array
        input 3D image.
    box_size : tuple of ints, optional
        size of box_size used for block decomposition. The default is (32, 32).
    filter_size : tuple of ints, optional
        Size of gaussian filter used for smoothing results. The default is (3, 3).

    Returns
    -------
    output : numpy array
        processed 3D image.

    """
    # TODO : refactor this with 2d version
    number_planes = image_3d.shape[0]
    output = np.zeros(image_3d.shape)
    sigma_clip = SigmaClip(sigma=3)
    bkg_estimator = MedianBackground()
    z_range = trange(number_planes)
    for z in z_range:
        image_2d = image_3d[z, :, :]
        bkg = Background2D(
            image_2d,
            box_size,
            filter_size=filter_size,
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
        )
        output[z, :, :] = image_2d - bkg.background
    return output


def combine_blocks_image_by_reprojection(
    block_ref, block_target, shift_matrices=None, axis1=0
):
    print(f"> Combining blocks into image by reprojection along axis {axis1}")
    number_blocks = block_ref.shape[0]
    block_sizes = list(block_ref.shape[2:])
    block_sizes.pop(axis1)
    img_sizes = [x * number_blocks for x in block_sizes]

    # gets ranges for slicing
    slice_coordinates = [
        [range(x * block_size, (x + 1) * block_size) for x in range(number_blocks)]
        for block_size in block_sizes
    ]

    # creates output images
    shifted_blocks_by_axis = np.zeros((img_sizes[0], img_sizes[1], 3))
    mse_as_blocks = np.zeros((number_blocks, number_blocks))
    nrmse_as_blocks = np.zeros((number_blocks, number_blocks))

    # blank image for blue channel to show borders between blocks
    blue = np.zeros(block_sizes)
    blue[0, :], blue[:, 0], blue[:, -1], blue[-1, :] = [0.5] * 4

    # reassembles image
    # takes one plane block
    for i, i_slice in enumerate(tqdm(slice_coordinates[0])):
        for j, j_slice in enumerate(slice_coordinates[1]):
            imgs = [block_ref[i, j]]
            if shift_matrices is not None:
                shift_3d = np.array(
                    [x[i, j] for x in shift_matrices]
                )  # gets 3D shift from block decomposition
                imgs.append(
                    shift_image(block_target[i, j], shift_3d)
                )  # realigns and appends to image list
            else:
                # appends original target with no re-alignment
                imgs.append(block_target[i, j])
            # projects along axis1
            imgs = [np.sum(x, axis=axis1) for x in imgs]
            # rescales intensity values
            imgs = [exposure.rescale_intensity(x, out_range=(0, 1)) for x in imgs]
            # adjusts pixel intensities
            imgs = [
                image_adjust(x, lower_threshold=0.5, higher_threshold=0.9999)
                for x in imgs
            ]
            mse_as_blocks[i, j] = mean_squared_error(imgs[0], imgs[1])
            nrmse_as_blocks[i, j] = normalized_root_mse(
                imgs[0], imgs[1], normalization="euclidean"
            )
            # appends last channel with grid
            imgs.append(blue)
            # makes block rgb image
            rgb = np.dstack(imgs)
            # inserts block into final rgb stack
            shifted_blocks_by_axis[
                i_slice[0] : i_slice[-1] + 1, j_slice[0] : j_slice[-1] + 1, :
            ] = rgb

    return shifted_blocks_by_axis, mse_as_blocks, nrmse_as_blocks


def heatmap(
    data,
    row_labels,
    col_labels,
    ax=None,
    cbar_kw=None,
    cbarlabel="",
    fontsize=12,
    **kwargs,
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", size=fontsize)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, size=fontsize)
    ax.set_yticklabels(row_labels, size=fontsize)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for _, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.1f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):  # sourcery skip: dict-assign-update-to-union
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    new_kwargs = dict(horizontalalignment="center", verticalalignment="center")
    new_kwargs.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            new_kwargs.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            im.axes.text(j, i, valfmt(data[i, j], None), **new_kwargs)
