from tifffile import imread, imsave

from .main import get_img_name, get_file_path


class Module:
    def __init__(
        self, supplementary_data: dict = None, action_keyword: str = None
    ) -> None:
        self.supplementary_data = supplementary_data
        self.action_keyword = action_keyword

    def run(self, data: Any):
        raise NotImplementedError

    def load_data(self, input_path, label_name):
        raise NotImplementedError

    def load_supplementary_data(self, input_path, label_name):
        raise NotImplementedError

    def save_data(self, output_path, label_name, data):
        raise NotImplementedError


class TiffModule(Module):
    def __init__(self, action_keyword: str) -> None:
        super().__init__(action_keyword=action_keyword)

    def run(self, array_3d):
        raise NotImplementedError

    def load_data(self, input_path, label_name):
        img_name = get_img_name(label_name)
        image_path = get_file_path(input_path, img_name, "tif")
        image = imread(image_path)
        return image

    def save_data(self, output_path, label_name, data):
        img_name = get_img_name(label_name)
        imsave(
            get_file_path(output_path, img_name + "_" + self.action_keyword, "tif"),
            data,
        )
