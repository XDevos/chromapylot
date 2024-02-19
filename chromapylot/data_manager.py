class DataManager:
    @staticmethod
    def read_image(image_path):
        return cv2.imread(image_path)

    @staticmethod
    def read_npy(input_path):
        return np.load(input_path)

    @staticmethod
    def read_csv(input_path):
        return pd.read_csv(input_path)

    @staticmethod
    def write_image(image, output_path):
        cv2.imwrite(output_path, image)

    @staticmethod
    def write_npy(array, output_path):
        np.save(output_path, array)

    @staticmethod
    def write_csv(data, output_path):
        data.to_csv(output_path)
