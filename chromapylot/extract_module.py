from skimage import measure
from astropy.table import Table
from tifffile import imread
import numpy as np

from .main import get_file_path
from .module import Module


def extract_properties(tiff_image, mask_3d):
    # Measure properties of labeled image regions
    properties = measure.regionprops_table(
        mask_3d,
        intensity_image=tiff_image,
        properties=(
            "label",
            "max_intensity",
        ),
    )
    return properties


class ExtractModule(Module):
    def __init__(self) -> None:
        super().__init__(supplementary_data={"_shifted": None})

    def run(self, tiff_image, mask_3d):
        properties = extract_properties(tiff_image, mask_3d)
        return Table(properties)

    def load_supplementary_data(self, input_path):
        shifted_img_path = get_file_path(input_path, "_shifted", "tif")
        self.supplementary_data["_shifted"] = imread(shifted_img_path)

    def load_data(self, input_path):
        mask_path = get_file_path(input_path, "_3Dmasks", "npy")
        masks = np.load(mask_path)
        return masks

    def save_data(self, output_path, data):
        data.write(
            get_file_path(output_path, "_props", "ecsv"),
            format="ascii.ecsv",
            overwrite=True,
        )
