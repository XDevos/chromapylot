from tifffile import imread, imsave

from .main import get_img_name, get_file_path


class Module:
    def __init__(self, supplementary_data: dict = None) -> None:
        self.supplementary_data = supplementary_data

    def run(self, tiff_image, mask_3d):
        raise NotImplementedError

    def load_data(self, input_path, label_name):
        raise NotImplementedError
    
    def load_supplementary_data(self, input_path, label_name):
        raise NotImplementedError

    def save_data(self, output_path, label_name, data):
        raise NotImplementedError

class TiffModule(Module):
    def __init__(self, action_keyword:str) -> None:
        super().__init__()
        self.action_keyword = action_keyword

    def run(self, array_3d):
        raise NotImplementedError

    def load_data(self, input_path, label_name):
        img_name = get_img_name(label_name)
        image_path = get_file_path(input_path, img_name, "tif")
        image = imread(image_path)
        return image

    def save_data(self, output_path, label_name, data):
        img_name = get_img_name(label_name)
        imsave(get_file_path(output_path, img_name + "_" + self.action_keyword, "tif"), data)


# class PreprocessingModule:
#     def process(self, input_data):
#         # Réduire le bruit
#         # Améliorer le contraste
#         return input_data

#     def load_input(self, input_path):
#         if input_path.endswith(".npy"):
#             return DataManager.read_npy(input_path)
#         elif input_path.endswith(".csv"):
#             return DataManager.read_csv(input_path)
#         else:  # Assume it's an image
#             return DataManager.read_image(input_path)

#     def save_output(self, output, output_path):
#         if isinstance(output, np.ndarray):
#             DataManager.write_npy(output, output_path)
#         elif isinstance(output, pd.DataFrame):
#             DataManager.write_csv(output, output_path)
#         else:  # Assume it's an image
#             DataManager.write_image(output, output_path)


# class AlignmentModule:
#     def __init__(self, method="method1"):
#         if method not in ["method1", "method2"]:
#             raise ValueError(f"Invalid alignment method: {method}")
#         self.method = method

#     def process(self, image):
#         if self.method == "method1":
#             return self.align_method1(image)
#         elif self.method == "method2":
#             return self.align_method2(image)

#     def align_method1(self, image):
#         # Implémenter l'alignement de la méthode 1*
#         return image

#     def align_method2(self, image):
#         # Implémenter l'alignement de la méthode 2*
#         return image

#     def load_input(self, input_path):
#         if input_path.endswith(".npy"):
#             return DataManager.read_npy(input_path)
#         elif input_path.endswith(".csv"):
#             return DataManager.read_csv(input_path)
#         else:  # Assume it's an image
#             return DataManager.read_image(input_path)

#     def save_output(self, output, output_path):
#         if isinstance(output, np.ndarray):
#             DataManager.write_npy(output, output_path)
#         elif isinstance(output, pd.DataFrame):
#             DataManager.write_csv(output, output_path)
#         else:  # Assume it's an image
#             DataManager.write_image(output, output_path)
