#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib

matplotlib.use("Agg")

import numpy as np
import scipy.optimize as spo
from scipy.stats import sigmaclip
from skimage import filters, io
from skimage.util.shape import view_as_blocks
from tqdm import trange
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib import ticker

from chromapylot.core.core_types import DataType
from chromapylot.core.data_manager import save_npy
from chromapylot.parameters.projection_params import ProjectionParams
from chromapylot.modules.module import Module
from chromapylot.core.data_manager import DataManager

from datetime import datetime
from chromapylot.core.data_manager import load_json
from chromapylot.core.run_args import RunArgs
from chromapylot.parameters.pipeline_params import PipelineParams
from chromapylot.core.core_types import AnalysisType


class ProjectModule(Module):
    def __init__(self, projection_params: ProjectionParams):
        super().__init__(input_type=DataType.IMAGE_3D, output_type=DataType.IMAGE_2D)
        self.supplementary_type = "cycle"
        self.mode = projection_params.mode
        self.dirname = "project"
        self.block_size = projection_params.block_size
        self.zwindows = projection_params.zwindows
        self.focal_plane_matrix = {}
        self.focus_plane = {}

    def load_data(self, input_path):
        print(f"[Loading] 3D image from {input_path}")
        return io.imread(input_path).squeeze()

    def save_data(self, data, output_dir, input_path):
        base = os.path.basename(input_path).split(".")[0]
        npy_filename = base + "_2d.npy"
        npy_path = os.path.join(output_dir, self.dirname, "data", npy_filename)
        save_npy(data, npy_path)
        png_filename = base + "_2d.png"
        png_path = os.path.join(output_dir, self.dirname, png_filename)
        self._save_png(data, png_path)
        if self.mode == "laplacian":
            cycle = DataManager.get_cycle_from_path(input_path)
            print(f"Saving focal plane for cycle {cycle}")
            focal_filename = base + "_focalPlaneMatrix.png"
            focal_path = os.path.join(output_dir, self.dirname, focal_filename)
            self._save_focal_plane(focal_path, cycle)
        else:
            raise NotImplementedError

    def run(self, array_3d, cycle: str = None):
        if self.mode == "laplacian":
            img_projected = self._projection_laplacian(array_3d, cycle)
            return img_projected
        else:
            raise NotImplementedError

    def _projection_laplacian(self, img, cycle: str):
        focal_plane_matrix, z_range, block = reinterpolate_focal_plane(
            img, block_size_xy=self.block_size, window=self.zwindows
        )
        output = reassemble_images(focal_plane_matrix, block, window=self.zwindows)

        self.focal_plane_matrix[cycle] = focal_plane_matrix
        self.focus_plane[cycle] = z_range[0]
        return output

    def _save_png(self, data, output_path):
        fig = plt.figure()
        size = (10, 10)
        fig.set_size_inches(size)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        norm = ImageNormalize(stretch=SqrtStretch())
        ax.set_title("2D Data")
        fig.add_axes(ax)
        ax.imshow(data, origin="lower", cmap="Greys_r", norm=norm)
        fig.savefig(output_path)
        plt.close(fig)

    def _save_focal_plane(self, output_path, cycle: str):
        cbarlabels = ["focalPlane"]
        fig, axes = plt.subplots(1, 1)
        fig.set_size_inches((2, 5))
        fig.suptitle(f"focal plane = {self.focus_plane[cycle]:.2f}")
        cbar_kw = {"fraction": 0.046, "pad": 0.04}

        ax = axes
        # image_show_with_values_single(
        #     ax, self.focal_plane_matrix[cycle], "focalPlane", 6, cbar_kw
        # )
        row = [str(x) for x in range(self.focal_plane_matrix[cycle].shape[0])]
        im, _ = heatmap(
            self.focal_plane_matrix[cycle],
            row,
            row,
            ax=ax,
            cmap="YlGn",
            cbarlabel="focalPlane",
            fontsize=6,
            cbar_kw=cbar_kw,
        )
        _ = annotate_heatmap(
            im,
            valfmt="{x:.0f}",
            size=6,
            threshold=None,
            textcolors=("black", "white"),
        )

        fig.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        self.focal_plane_matrix[cycle] = None
        self.focus_plane[cycle] = None


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
    # Check if max(laplacian_variance) is zero
    if np.max(laplacian_variance) != 0:
        laplacian_variance = laplacian_variance / np.max(laplacian_variance)
    else:
        # Handle the case when max(laplacian_variance) is zero
        laplacian_variance = np.zeros_like(laplacian_variance)

    # If laplacian_variance contains NaN values
    if np.isnan(laplacian_variance).any():
        # Handle or remove NaN values
        laplacian_variance = np.nan_to_num(laplacian_variance)
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
    texts = []
    data = data.filled(np.nan)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            new_kwargs.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **new_kwargs)
            texts.append(text)

    return texts


def main(command_line_args=None):
    begin_time = datetime.now()
    print(f"Start time: {begin_time}")

    # INITIALIZATION
    run_args = RunArgs(command_line_args)
    raw_params = load_json(os.path.join(run_args.input, "parameters.json"))
    pipe_params = PipelineParams(raw_params, AnalysisType.TRACE)
    mod = ProjectModule(pipe_params.projection)

    # MODULE EXECUTION
    input_data = mod.load_data(run_args.in_file)
    cycle = DataManager.get_cycle_from_path(run_args.in_file)
    output_data = mod.run(input_data, cycle)
    mod.save_data(output_data, run_args.output, run_args.in_file)

    # TERMINATION
    print("\n==================== Normal termination ====================\n")
    print(f"Elapsed time: {datetime.now() - begin_time}")


if __name__ == "__main__":
    main()
