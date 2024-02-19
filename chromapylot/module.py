class PreprocessingModule:
    def process(self, input_data):
        # Réduire le bruit
        # Améliorer le contraste
        return input_data

    def load_input(self, input_path):
        if input_path.endswith(".npy"):
            return DataManager.read_npy(input_path)
        elif input_path.endswith(".csv"):
            return DataManager.read_csv(input_path)
        else:  # Assume it's an image
            return DataManager.read_image(input_path)

    def save_output(self, output, output_path):
        if isinstance(output, np.ndarray):
            DataManager.write_npy(output, output_path)
        elif isinstance(output, pd.DataFrame):
            DataManager.write_csv(output, output_path)
        else:  # Assume it's an image
            DataManager.write_image(output, output_path)


class AlignmentModule:
    def __init__(self, method="method1"):
        if method not in ["method1", "method2"]:
            raise ValueError(f"Invalid alignment method: {method}")
        self.method = method

    def process(self, image):
        if self.method == "method1":
            return self.align_method1(image)
        elif self.method == "method2":
            return self.align_method2(image)

    def align_method1(self, image):
        # Implémenter l'alignement de la méthode 1*
        return image

    def align_method2(self, image):
        # Implémenter l'alignement de la méthode 2*
        return image

    def load_input(self, input_path):
        if input_path.endswith(".npy"):
            return DataManager.read_npy(input_path)
        elif input_path.endswith(".csv"):
            return DataManager.read_csv(input_path)
        else:  # Assume it's an image
            return DataManager.read_image(input_path)

    def save_output(self, output, output_path):
        if isinstance(output, np.ndarray):
            DataManager.write_npy(output, output_path)
        elif isinstance(output, pd.DataFrame):
            DataManager.write_csv(output, output_path)
        else:  # Assume it's an image
            DataManager.write_image(output, output_path)
