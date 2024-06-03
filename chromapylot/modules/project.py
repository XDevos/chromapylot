#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib

matplotlib.use("Agg")

import numpy as np
import scipy.optimize as spo
from scipy.stats import sigmaclip
from skimage import filters
from skimage.util.shape import view_as_blocks
from tqdm import trange
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib import ticker

from chromapylot.core.core_types import DataType
from chromapylot.core.data_manager import save_npy
from chromapylot.parameters.projection_params import ProjectionParams
from chromapylot.modules.routine import Module
from chromapylot.core.data_manager import DataManager, create_npy_path, create_png_path

from datetime import datetime
from chromapylot.core.data_manager import load_json
from chromapylot.core.run_args import RunArgs
from chromapylot.parameters.params_manager import ParamsManager
from chromapylot.core.core_types import AnalysisType

from apifish.stack import projection


class ProjectModule(Module):
    def __init__(
        self,
        data_manager: DataManager,
        projection_params: ProjectionParams,
        input_type: DataType = DataType.IMAGE_3D,
        output_type: DataType = DataType.IMAGE_2D,
        supplementary_type: DataType = None,
    ):
        super().__init__(
            data_manager=data_manager,
            input_type=input_type,
            output_type=output_type,
            supplementary_type=supplementary_type,
        )
        self.dirname = "project"
        self.mode = projection_params.mode
        self.block_size = projection_params.block_size
        self.zwindows = projection_params.zwindows
        self.window_security = projection_params.window_security
        self.z_project_option = projection_params.z_project_option
        self.zmin = projection_params.zmin
        self.zmax = projection_params.zmax

    def load_data(self, input_path):
        return self.data_m.load_image_3d(input_path)

    def save_data(self, data, input_path, input_data, supplementary_data):
        print("[Save] 2D npy | 2D png")
        npy_path = create_npy_path(
            input_path, self.data_m.output_folder, self.dirname, "_2d"
        )
        save_npy(data, npy_path, self.data_m.out_dir_len)
        png_path = create_png_path(
            input_path, self.data_m.output_folder, self.dirname, "_2d"
        )
        save_png(data, png_path, self.data_m.out_dir_len)

    def run(self, array_3d, cycle: str = None):
        print(f"[Run] {self.input_type.value} -> {self.output_type.value}")
        if self.mode == "laplacian":
            raise ValueError(
                "Laplacian projection is implemented with SplitInBlocks + InterpolateFocalPlane + ProjectByBlockModule modules."
            )
        elif self.mode == "automatic":
            zmin, zmax = self._precise_z_planes_auto(array_3d)
            return self.projection_2d(array_3d[zmin : zmax + 1])
        elif self.mode == "full":
            return self.projection_2d(array_3d)
        elif self.mode == "manual":
            zmin, zmax = self.check_zmin_zmax(array_3d.shape[0])
            # TODO: update data for regression test to accept [zmin : zmax + 1]
            # return self.projection_2d(array_3d[zmin : zmax + 1])
            return self.projection_2d(array_3d[zmin:zmax])
        else:
            raise ValueError(
                f"Projection mode UNRECOGNIZED: {self.mode}\n> Available mode: automatic,full,manual,laplacian"
            )

    def _precise_z_planes_auto(self, img):
        """
        Calculates the focal planes based max standard deviation
        Finds best focal plane by determining the max of the std deviation vs z curve
        """
        win_sec = self.window_security
        print(f"win_sec: {win_sec}")
        print(f"zmin: {self.zmin}")
        print(f"zmax: {self.zmax}")
        zmin, zmax = self.check_zmin_zmax(img.shape[0])

        print(f"zmin: {zmin}")
        print(f"zmax: {zmax}")
        nb_of_planes = zmax - zmin
        print(f"nb_of_planes: {nb_of_planes}")
        std_matrix = np.zeros(nb_of_planes)
        mean_matrix = np.zeros(nb_of_planes)

        # calculate STD in each plane
        for i in range(nb_of_planes):
            std_matrix[i] = np.std(img[i])
            mean_matrix[i] = np.mean(img[i])

        max_std = np.max(std_matrix)
        i_focus_plane = np.where(std_matrix == max_std)[0][0]

        # Select a window to avoid being on the edges of the stack
        if i_focus_plane < win_sec or (i_focus_plane > nb_of_planes - win_sec):
            focus_plane = i_focus_plane
        else:
            # interpolate zfocus
            axis_z = range(
                max(
                    zmin,
                    i_focus_plane - win_sec,
                    min(zmax, i_focus_plane + win_sec),
                )
            )
            std_matrix -= np.min(std_matrix)
            std_matrix /= np.max(std_matrix)

            try:
                print(f"axis_z: {axis_z}")
                print(f"len std_matrix: {len(std_matrix)}")
                fitgauss = spo.curve_fit(
                    projection.gaussian, axis_z, std_matrix[axis_z[0] : axis_z[-1] + 1]
                )
                focus_plane = int(fitgauss[0][1])
            except RuntimeError:
                print("Warning, too many iterations")
                focus_plane = i_focus_plane

        zmin = max(win_sec, focus_plane - self.zwindows)
        zmax = min(nb_of_planes, win_sec + nb_of_planes, focus_plane + self.zwindows)

        return zmin, zmax

    def projection_2d(self, img):
        if self.z_project_option == "MIP":
            return img.max(axis=0)
        elif self.z_project_option == "sum":
            return projection.sum_projection(img)
        else:
            print(
                f"ERROR: option not recognized. Expected: MIP or sum. Read: {self.z_project_option}"
            )

    def check_zmin_zmax(self, n_planes):
        zmin = self.zmin
        zmax = self.zmax
        if self.zmin < 0:
            print(f"zmin < 0: {self.zmin}")
            zmin = 0
        if self.zmax > n_planes:
            print(f"zmax > n_planes: {self.zmax} > {n_planes}")
            zmax = n_planes
        if self.zmin > zmax:
            print(f"zmin > zmax: {self.zmin} > {self.zmax}")
            zmin = zmax
            print(f"zmin set to zmax: {zmin}")
        return zmin, zmax


class SplitInBlocks(ProjectModule):
    def __init__(self, data_manager: DataManager, projection_params: ProjectionParams):
        super().__init__(
            data_manager=data_manager,
            projection_params=projection_params,
            output_type=DataType.IMAGE_BLOCKS,
        )

    def run(self, img):
        return split_in_blocks(img, block_size_xy=self.block_size)

    def save_data(self, data, input_path, input_data, supplementary_data):
        print("> No need to save data for SplitInBlocks module.")
        pass


class InterpolateFocalPlane(ProjectModule):
    def __init__(self, data_manager: DataManager, projection_params: ProjectionParams):
        super().__init__(
            data_manager=data_manager,
            projection_params=projection_params,
            input_type=DataType.IMAGE_BLOCKS,
            output_type=DataType.MATRIX_2D,
        )

    def run(self, blocks):
        return calculate_focus_per_block(blocks)

    def load_data(self, input_path):
        print(f"[Load] {self.input_type.value}")
        short_path = input_path[self.data_m.in_dir_len :]
        print(f"> $INPUT{short_path}")
        return np.load(input_path)

    def save_data(self, data, in_path, input_data, supplementary_data):
        output_path = create_png_path(
            in_path, self.data_m.output_folder, self.dirname, "_focalPlaneMatrix"
        )
        fig, axes = plt.subplots(1, 1)
        fig.set_size_inches((2, 5))
        focus_plane = get_focus_plane(data)
        fig.suptitle(f"focal plane = {focus_plane:.2f}")
        cbar_kw = {"fraction": 0.046, "pad": 0.04}

        ax = axes
        row = [str(x) for x in range(data.shape[0])]
        im, _ = heatmap(
            data,
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
        short_path = output_path[self.data_m.out_dir_len :]
        print(f"> $OUTPUT{short_path}")


class ProjectByBlockModule(ProjectModule):
    def __init__(self, data_manager: DataManager, projection_params: ProjectionParams):
        super().__init__(
            data_manager=data_manager,
            projection_params=projection_params,
            input_type=DataType.MATRIX_2D,
            supplementary_type=DataType.IMAGE_BLOCKS,
        )

    def run(self, focal_matrix, blocks):
        return reassemble_images(focal_matrix, blocks, window=self.zwindows)

    def load_data(self, input_path):
        print(f"[Load] {self.input_type.value}")
        short_path = input_path[self.data_m.in_dir_len :]
        print(f"> $INPUT{short_path}")
        return np.load(input_path)

    # =============================================================================
    # FOCAL PLANE INTERPOLATION
    # =============================================================================


def save_png(data, output_path, len_out_dir):
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
    short_path = output_path[len_out_dir:]
    print(f"> $OUTPUT{short_path}")


def split_in_blocks(data, block_size_xy=256):
    n_z_planes = data.shape[0]
    block_size = (n_z_planes, block_size_xy, block_size_xy)
    block = view_as_blocks(data, block_size)
    # block.shape ~= (n_z_blocks, n_x_blocks, n_y_blocks, n_z_planes, 128, 128)
    # But n_z_blocks = 1 due to we give the full z stack to the function
    block = block.squeeze()  # So we remove the first dimension
    return block


def get_focus_plane(focal_matrix):
    focal_planes_to_process = focal_matrix[~np.isnan(focal_matrix)]
    focal_plane, _, _ = sigmaclip(focal_planes_to_process, high=3, low=3)
    focus_plane = np.mean(focal_plane)
    if np.isnan(focus_plane):
        raise ValueError("Focus plane detection failed.")
    else:
        focus_plane = np.mean(focal_plane).astype("int64")
    return focus_plane


def calculate_focus_per_block(blocks):
    """
    Calculates the most likely focal plane of an image by breaking into blocks and calculating
    the focal plane in each block

    - breaks image into blocks
    - returns the focal plane + fwhm for each block

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    Returns
    -------
    focal_plane_matrix: np array
        matrix containing the maximum of the laplacian variance per block
    """
    focal_plane_matrix = np.zeros(blocks.shape[0:2])  # (n_x_blocks, n_y_blocks)
    for i in trange(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            focal_plane_matrix[i, j] = find_focal_plane(blocks[i, j])
    return focal_plane_matrix


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
    raw_params = load_json(os.path.join(run_args.input, "infoList.json"))
    pipe_params = ParamsManager(raw_params, AnalysisType.TRACE)
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
