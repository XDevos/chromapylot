from tifffile import imread, imsave
from typing import Any

from .main import get_img_name, get_file_path
from .chromapylot import DataType


class Module:
    def __init__(
        self,
        input_type: DataType,
        output_type: DataType,
        reference_type: DataType = None,
        supplementary_type: DataType = None,
    ):
        self.input_type = input_type
        self.output_type = output_type
        self.reference_type = reference_type
        self.reference_data = None
        self.supplementary_type = supplementary_type

    def run(self, data: Any):
        raise NotImplementedError

    def load_data(self, input_path):
        raise NotImplementedError

    def load_supplementary_data(self, input_path):
        raise NotImplementedError

    def save_data(self, output_path, data):
        raise NotImplementedError


class TiffModule(Module):
    def __init__(self, output_type: str) -> None:
        super().__init__(output_type=output_type)

    def run(self, array_3d):
        raise NotImplementedError

    def load_data(self, input_path):
        image_path = get_file_path(input_path, "", "tif")
        image = imread(image_path)
        return image

    def save_data(self, output_path, data):
        imsave(get_file_path(output_path, "_" + self.output_type, "tif"), data)
