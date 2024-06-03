#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
import time
import os

from stardist.models import StarDist3D
from csbdeep.data import PadAndCropResizer
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization.stretch import SqrtStretch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
from astropy.stats import SigmaClip, gaussian_fwhm_to_sigma
from photutils import (
    Background2D,
    MedianBackground,
    deblend_sources,
    detect_sources,
    detect_threshold,
)
from csbdeep.utils.tf import limit_gpu_memory
from astropy.convolution import Gaussian2DKernel, convolve
from skimage.measure import regionprops
from scipy.spatial import Voronoi
from matplotlib.path import Path
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from stardist import random_label_cmap
from skimage.util.apply_parallel import apply_parallel
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from tqdm import trange
from skimage import measure

from chromapylot.core.core_types import DataType
from chromapylot.core.data_manager import (
    DataManager,
    create_png_path,
    create_npy_path,
    save_npy,
)
from chromapylot.routines.routine import Module
from chromapylot.parameters.segmentation_params import SegmentationParams


class Segment2D(Module):
    def __init__(
        self,
        data_manager: DataManager,
        segmentation_params: SegmentationParams,
    ):
        super().__init__(
            data_manager=data_manager,
            input_type=DataType.IMAGE_2D_SHIFTED,
            output_type=DataType.SEGMENTED_2D,
            reference_type=None,
            supplementary_type=None,
        )
        self.dirname = "mask_2d"
        self.background_method = segmentation_params.background_method
        self.tesselation = segmentation_params.tesselation
        self.background_sigma = segmentation_params.background_sigma
        self.threshold_over_std = segmentation_params.threshold_over_std
        self.area_min = segmentation_params.area_min
        self.area_max = segmentation_params.area_max
        self.fwhm = segmentation_params.fwhm
        self.stardist_basename = segmentation_params.stardist_basename
        self.stardist_network = segmentation_params.stardist_network

    def load_data(self, input_path):
        return self.data_m.load_image_2d(input_path)

    def load_reference_data(self, paths: List[str]):
        raise NotImplementedError

    def load_supplementary_data(self, input_path, cycle):
        raise NotImplementedError

    def run(self, data, supplementary_data=None):
        if self.background_method == "inhomogeneous":
            output = self.segment_mask_inhomog_background(data)
            if self.tesselation:
                output = tessellate_masks(output)
        elif self.background_method == "stardist":
            output = self.segment_mask_stardist(data)
        else:
            raise ValueError(
                f"Segmentation method {self.background_method} not recognized, only 'inhomogeneous' and 'stardist' are supported for segment_2d"
            )
        return output

    def save_data(self, data, input_path, input_data, supplementary_data):
        postfix = "_segmentedMasks"
        if self.background_method == "stardist":
            postfix = "_stardist" + postfix
        png_path = create_png_path(
            input_path, self.data_m.output_folder, self.dirname, postfix
        )
        show_image_masks(
            input_data,
            data,
            png_path,
        )
        npy_path = create_npy_path(
            input_path, self.data_m.output_folder, self.dirname, "_Masks"
        )
        save_npy(data, npy_path, self.data_m.out_dir_len)

    def segment_mask_inhomog_background(self, im):
        """
        Function used for segmenting masks with the ASTROPY library that uses image processing

        Parameters
        ----------
        im : 2D np array
            image to be segmented.

        Returns
        -------
        segm_deblend: 2D np array where each pixel contains the label of the mask segmented. Background: 0

        """
        # removes background
        threshold = detect_threshold(im, nsigma=2.0)
        sigma_clip = SigmaClip(sigma=self.background_sigma)

        bkg_estimator = MedianBackground()
        bkg = Background2D(
            im,
            (64, 64),
            filter_size=(3, 3),
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
        )
        threshold = bkg.background + (
            self.threshold_over_std * bkg.background_rms
        )  # background-only error image, typically 1.0

        sigma = self.fwhm * gaussian_fwhm_to_sigma  # FWHM = 3.
        kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
        kernel.normalize()
        data = convolve(im, kernel, mask=None, normalize_kernel=True)
        # estimates masks and deblends
        segm = detect_sources(data, threshold, npixels=self.area_min)

        # removes masks too close to border
        segm.remove_border_labels(
            border_width=10
        )  # TODO: parameter to add to parameters.json ?

        segm_deblend = deblend_sources(
            data,
            segm,
            npixels=self.area_min,  # typically 50 for masks
            nlevels=32,
            contrast=0.001,  # try 0.2 or 0.3
            relabel=True,
        )

        # removes Masks too big or too small
        for label in segm_deblend.labels:
            # take regions with large enough areas
            area = segm_deblend.get_area(label)
            # print_log('label {}, with area {}'.format(label,area))
            if area < self.area_min or area > self.area_max:
                segm_deblend.remove_label(label=label)
                # print_log('label {} removed'.format(label))

        # relabel so masks numbers are consecutive
        segm_deblend.relabel_consecutive()

        return segm_deblend

    def segment_mask_stardist(self, im):
        """
        Function used for segmenting masks with the STARDIST package that uses Deep Convolutional Networks

        Parameters
        ----------
        im : 2D np array
            image to be segmented.

        Returns
        -------
        segm_deblend: 2D np array where each pixel contains the label of the mask segmented. Background: 0

        """

        np.random.seed(6)
        sigma = self.fwhm * gaussian_fwhm_to_sigma  # FWHM = 3.
        kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
        kernel.normalize()

        n_channel = 1 if im.ndim == 2 else im.shape[-1]
        axis_norm = (0, 1)  # normalize channels independently

        if n_channel > 1:
            print(
                f'> Normalizing image channels {"jointly" if axis_norm is None or 2 in axis_norm else "independently"}.'
            )
        if self.stardist_basename is not None and os.path.exists(
            self.stardist_basename
        ):
            base_dir = self.stardist_basename
        else:
            base_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                os.pardir,
                "stardist_models",
            )
        if self.stardist_network is not None and os.path.exists(
            os.path.join(base_dir, self.stardist_network)
        ):
            model_name = self.stardist_network
        else:
            model_name = "DAPI_2D_stardist_nc14_nrays64_epochs40_grid2"
        model = StarDist2D(None, name=model_name, basedir=base_dir)

        img = normalize(im, 1, 99.8, axis=axis_norm)
        labeled, _ = model.predict_instances(img)

        return labeled


class Segment3D(Module):
    def __init__(
        self,
        data_manager: DataManager,
        segmentation_params: SegmentationParams,
    ):
        super().__init__(
            data_manager=data_manager,
            input_type=DataType.IMAGE_3D_SHIFTED,
            output_type=DataType.SEGMENTED_3D,
            reference_type=None,
            supplementary_type=None,
        )
        self.dirname = "mask_3d"
        self.stardist_basename = segmentation_params.stardist_basename
        self.stardist_network3D = segmentation_params.stardist_network3D
        self.method_3d = segmentation_params._3Dmethod
        self.limit_x = segmentation_params.limit_x
        self._3D_sigma = segmentation_params._3D_sigma
        self._3D_threshold_over_std = segmentation_params._3D_threshold_over_std
        self._3D_area_min = segmentation_params._3D_area_min
        self._3D_area_max = segmentation_params._3D_area_max
        self._3D_nlevels = segmentation_params._3D_nlevels
        self._3D_contrast = segmentation_params._3D_contrast

    def load_data(self, input_path):
        return self.data_m.load_image_3d(input_path)

    def load_reference_data(self, paths: List[str]):
        raise NotImplementedError

    def load_supplementary_data(self, input_path, cycle):
        raise NotImplementedError

    def run(self, data, supplementary_data=None):
        if self.method_3d == "stardist":
            base_dir = self.find_base_dir()
            model_name = self.find_model_name(base_dir)
            print(f"> Loading model {model_name} from {base_dir} ...")
            labels = segment_3d_by_stardist(
                model_name, base_dir, data, limit_x=self.limit_x
            )
        elif self.method_3d == "thresholding":
            labels = segment_3d_by_thresholding(
                data,
                self._3D_sigma,
                self._3D_threshold_over_std,
                self._3D_area_min,
                self._3D_area_max,
                self._3D_nlevels,
                self._3D_contrast,
            )
        return labels

    def save_data(self, segmented_image_3d, input_path, input_data, supplementary_data):
        number_masks = np.max(segmented_image_3d)
        print(f"$ Number of masks detected: {number_masks}")

        npy_2d_path = create_npy_path(
            input_path, self.data_m.output_folder, self.dirname, "_Masks"
        )
        npy_3d_path = create_npy_path(
            input_path, self.data_m.output_folder, self.dirname, "_3Dmasks"
        )

        # saves 3D image

        save_npy(segmented_image_3d, npy_3d_path, self.data_m.out_dir_len)

        # saves 2D image
        segmented_image_2d = np.max(segmented_image_3d, axis=0)
        save_npy(segmented_image_2d, npy_2d_path, self.data_m.out_dir_len)

        png_path = create_png_path(
            input_path, self.data_m.output_folder, self.dirname, ".tif_3Dmasks"
        )
        plot_raw_images_and_labels(input_data, segmented_image_3d, png_path)

    def find_base_dir(self):
        if self.stardist_basename is not None and os.path.exists(
            self.stardist_basename
        ):
            base_dir = self.stardist_basename
        else:
            base_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                os.pardir,
                "stardist_models",
            )
        return base_dir

    def find_model_name(self, base_dir):
        if self.stardist_network3D is not None and os.path.exists(
            os.path.join(base_dir, self.stardist_network3D)
        ):
            model_name = self.stardist_network3D
        else:
            model_name = "DAPI_3D_stardist_17032021_deconvolved"
        return model_name


class Deblend3D(Module):
    def __init__(
        self,
        data_manager: DataManager,
        segmentation_params: SegmentationParams,
    ):
        super().__init__(
            data_manager=data_manager,
            input_type=DataType.SEGMENTED_3D,
            output_type=DataType.SEGMENTED_3D,
            reference_type=None,
            supplementary_type=None,
        )
        self.dirname = "mask_3d"

    def run(self, labels):
        binary = np.array(labels > 0, dtype=int)
        distance = apply_parallel(ndi.distance_transform_edt, binary)
        print(" > Deblending sources in 3D by watersheding...")
        coords = peak_local_max(
            distance, footprint=np.ones((10, 10, 25)), labels=binary
        )
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=binary)
        return labels

    def save_data(self, data, input_path, input_data, supplementary_data):
        pass


def tessellate_masks(segm_deblend):
    """
    * takes a labeled mask (background 0, nuclei labeled 1, 2, ...)
    * calls get_tessellation(xy, img_shape)
    * returns the tesselated mask and the voronoi data structure

    Parameters
    ----------
    segm_deblend : TYPE
        Labeled mask.

    Returns
    -------
    voronoi_data : TYPE
        DESCRIPTION.
    mask_voronoi : TYPE
        DESCRIPTION.

    """
    start_time = time.time()

    # get centroids
    mask_labeled = segm_deblend.data
    mask_binary = mask_labeled.copy()
    mask_binary[mask_binary > 0] = 1

    regions = regionprops(mask_labeled)

    num_masks = np.max(mask_labeled)
    centroid = np.zeros((num_masks + 1, 2))  # +1 as labels run from 0 to max

    for props in regions:
        y_0, x_0 = props.centroid
        label = props.label
        centroid[label, :] = x_0, y_0

    # tesselation
    # remove first centroid (this is the background label)
    xy = centroid[1:, :]
    voronoi_data = get_tessellation(xy, mask_labeled.shape)

    # add some clipping to the tessellation
    # gaussian blur and thresholding; magic numbers!
    mask_blurred = gaussian_filter(mask_binary.astype("float64"), sigma=20)
    mask_blurred = mask_blurred > 0.01

    # convert tessellation to mask
    np.random.seed(42)

    mask_voronoi = np.zeros(mask_labeled.shape, dtype="int64")

    nx, ny = mask_labeled.shape

    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T

    # currently takes 1min for approx 600 polygons
    for label in range(num_masks):  # label is shifted by -1 now
        mask_id = label + 1

        idx_vor_region = voronoi_data.point_region[label]
        idx_vor_vertices = voronoi_data.regions[
            idx_vor_region
        ]  # list of indices of the Voronoi vertices

        vertices = np.full((len(idx_vor_vertices), 2), np.NaN)
        drop_vert = False
        # pylint: disable-next=consider-using-enumerate
        for i in range(len(idx_vor_vertices)):
            idx = idx_vor_vertices[i]
            if (
                idx == -1
            ):  # this means a "virtual point" at infinity as the vertex is not closed
                drop_vert = True
                print('$ Detected "virtual point" at infinity. Skipping this mask.')
                break
            vertices[i, :] = voronoi_data.vertices[idx]

        if drop_vert:  # region is not bounded
            continue

        poly_path = Path(vertices)
        mask = poly_path.contains_points(points)
        mask = mask.reshape((ny, nx))
        mask_voronoi[mask & mask_blurred] = mask_id

    # print_log("--- Took {:.2f}s seconds ---".format(time.time() - start_time))
    print(f"$ Tessellation took {(time.time() - start_time):.2f}s seconds.")

    return mask_voronoi


def get_tessellation(xy, img_shape):
    """
    * runs the actual tesselation based on the xy position of the markers in an image of given shape

    # follow this tutorial
    # https://hpaulkeeler.com/voronoi-dirichlet-tessellations/
    # https://github.com/hpaulkeeler/posts/blob/master/PoissonVoronoi/PoissonVoronoi.py

    # changes:
    # added dummy points outside of the image corners (in quite some distance)
    # they are supposed "catch" all the vertices that end up at infinity
    # follows an answer given here
    # https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram/20678647#20678647

    # Attributes
    #    points ndarray of double, shape (npoints, ndim)
    #        Coordinates of input points.
    #
    #    vertices ndarray of double, shape (nvertices, ndim)
    #        Coordinates of the Voronoi vertices.
    #
    #    ridge_points ndarray of ints, shape (nridges, 2)
    #        Indices of the points between which each Voronoi ridge lies.
    #
    #    ridge_vertices list of list of ints, shape (nridges, \*)
    #        Indices of the Voronoi vertices forming each Voronoi ridge.
    #
    #    regions list of list of ints, shape (nregions, \*)
    #        Indices of the Voronoi vertices forming each Voronoi region. -1 indicates vertex outside the Voronoi diagram.
    #
    #    point_region list of ints, shape (npoints)
    #        Index of the Voronoi region for each input point. If qhull option “Qc” was not specified, the list will contain -1 for points that are not associated with a Voronoi region.
    #
    #    furthest_site
    #        True if this was a furthest site triangulation and False if not.
    #        New in version 1.4.0.

    Parameters
    ----------
    xy : TYPE
        DESCRIPTION.
    img_shape : TYPE
        DESCRIPTION.

    Returns
    -------
    voronoi_data : TYPE
        DESCRIPTION.


    """

    x_center, y_center = np.array(img_shape) / 2
    x_max, y_max = np.array(img_shape)

    corner1 = [x_center - 100 * x_max, y_center - 100 * y_max]
    corner2 = [x_center + 100 * x_max, y_center - 100 * y_max]
    corner3 = [x_center - 100 * x_max, y_center + 100 * y_max]
    corner4 = [x_center + 100 * x_max, y_center + 100 * y_max]

    xy = np.append(xy, [corner1, corner2, corner3, corner4], axis=0)

    # perform Voroin tesseslation
    return Voronoi(xy)


def show_image_masks(im, segm_deblend, output_filename):
    lbl_cmap = random_label_cmap()
    norm = ImageNormalize(stretch=SqrtStretch())
    cmap = lbl_cmap
    fig = plt.figure()
    fig.set_size_inches((30, 30))
    plt.imshow(im, cmap="Greys_r", origin="lower", norm=norm)
    plt.imshow(segm_deblend, origin="lower", cmap=cmap, alpha=0.5)
    plt.savefig(output_filename)
    plt.close()


def plot_raw_images_and_labels(image, label, png_path):
    """
    Parameters
    ----------
    image : List of numpy ndarray (N-dimensional array)
        3D raw image of format .tif

    label : List of numpy ndarray (N-dimensional array)
        3D labeled image of format .tif
    """

    cmap = random_label_cmap()

    moy = np.mean(image, axis=0)
    lbl_moy = np.max(label, axis=0)

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches((50, 50))
    ax = axes.ravel()
    titles = ["raw image", "projected labeled image"]

    ax[0].imshow(moy, cmap="Greys_r", origin="lower")

    ax[1].imshow(lbl_moy, cmap=cmap, origin="lower")

    for axis, title in zip(ax, titles):
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(title)

    fig.savefig(png_path)


def segment_3d_by_stardist(model_name, base_dir, img_3d, limit_x=351):

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # why do we need this?

    model = StarDist3D(None, name=model_name, basedir=base_dir)
    limit_gpu_memory(None, allow_growth=True)

    im = normalize(img_3d, pmin=1, pmax=99.8, axis=(0, 1, 2))
    l_x = im.shape[1]
    if l_x < limit_x:  # what is this value? should it be a k-arg?
        labels, _ = model.predict_instances(im)
    else:
        resizer = PadAndCropResizer()
        axes = "ZYX"
        im = resizer.before(im, axes, model._axes_div_by(axes))
        labels, _ = model.predict_instances(im, n_tiles=(1, 8, 8))
        labels = resizer.after(labels, axes)

    return labels


def segment_3d_by_thresholding(
    image_3d, sigma, threshold_over_std, area_min, area_max, nlevels, contrast
):
    kernel = Gaussian2DKernel(sigma, x_size=sigma, y_size=sigma)
    kernel.normalize()
    output = np.zeros(image_3d.shape)
    number_planes = image_3d.shape[0]
    for z in trange(number_planes):
        image_2d = image_3d[z, :, :]
        image_2d_segmented = segment_2d_by_thresholding(
            image_2d,
            threshold_over_std=threshold_over_std,
            area_min=area_min,
            area_max=area_max,
            nlevels=nlevels,
            contrast=contrast,
            kernel=kernel,
        )
        output[z, :, :] = image_2d_segmented
    labels = measure.label(output)
    return labels


def segment_2d_by_thresholding(
    image_2d,
    threshold_over_std=10,
    area_min=3,
    area_max=1000,
    nlevels=64,
    contrast=0.001,
    kernel=None,
):
    # makes threshold matrix
    threshold = np.zeros(image_2d.shape)
    threshold[:] = threshold_over_std * image_2d.max() / 100
    if kernel is not None:
        data = convolve(image_2d, kernel, mask=None, normalize_kernel=True)
    # segments objects
    segm = detect_sources(
        data,
        threshold,
        npixels=area_min,
    )

    if segm.nlabels <= 0:
        # returns empty image as no objects were detected
        return segm.data
    # removes masks too close to border
    segm.remove_border_labels(border_width=10)
    if segm.nlabels <= 0:
        # returns empty image as no objects were detected
        return segm.data

    segm_deblend = deblend_sources(
        data,
        segm,
        npixels=area_min,  # watch out, this is per plane!
        nlevels=nlevels,
        contrast=contrast,
        relabel=True,
        mode="exponential",
    )
    if segm_deblend.nlabels > 0:
        # removes Masks too big or too small
        for label in segm_deblend.labels:
            # take regions with large enough areas
            area = segm_deblend.get_area(label)
            if area < area_min or area > area_max:
                segm_deblend.remove_label(label=label)

        # relabel so masks numbers are consecutive
        # segm_deblend.relabel_consecutive()

    # image_2d_segmented = segm.data % changed during recoding function
    image_2d_segmented = segm_deblend.data

    image_2d_segmented[image_2d_segmented > 0] = 1
    return image_2d_segmented
