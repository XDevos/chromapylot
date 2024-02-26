import os


def extract_files(root: str):
    """Extract recursively file informations of all files into a given directory.
    Note:
    * filepath is directory path with filename and extension
    * filename is the name without extension

    Parameters
    ----------
    root : str
        The name of root directory

    Returns
    -------
    List[Tuple(str,str,str)]
        List of file informations: (filepath, filename, extension)
    """
    files = []
    # Iterate into dirname and each subdirectories dirpath, dirnames, filenames
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            split_filename = filename.split(".")
            extension = split_filename.pop() if len(split_filename) > 1 else None
            short_filename = ".".join(split_filename)
            filepath = os.path.join(dirpath, filename)
            files.append((filepath, short_filename, extension))

        if len(dirnames) > 0:
            print(f"$ Inside {dirpath}, subdirectories detected:\n  {dirnames}")

    return files


class DataManager:
    def __init__(self, run_args):
        self.input_folder = run_args.input
        self.output_folder = run_args.output
        self.input_files = extract_files(self.input_folder)
        self.parameters_file = self.get_parameters_file()
        self.analysis_files = self.get_analysis_files()

    def get_parameters_file(self):
        for file in self.input_files:
            if file[1] == "parameters" and file[2] == "json":
                return file[0]
        raise FileNotFoundError(
            "No parameters file found in input folder: ", self.input_folder
        )

    def get_analysis_files(self):
        analysis_files = {
            "fiducial": [],
            "barcode": [],
            "dapi": [],
            "rna": [],
            "primer": [],
            "satellite": [],
            "trace": [],
        }
        for file in self.input_files:
            analysis_type = self.get_analysis_type(file[1], file[2])
            data_type = self.get_data_type(file[1], file[2])
            if analysis_type:
                analysis_files[analysis_type].append((data_type, file[0]))
        return analysis_files

    def get_analysis_type(self, filename, extension):
        if extension in ["tif", "tiff", "npy"]:
            cycle = self.get_cycle_from_path(filename)
            channel = self.get_channel_from_path(
                filename
            )  # TODO check ch depending parameters
            if cycle == "DAPI":
                if channel == "ch00":
                    return "dapi"
                elif channel == "ch01":
                    return "fiducial"
                elif channel == "ch02":
                    return "rna"
            elif "RT" in cycle:
                if channel == "ch00":
                    return "fiducial"
                elif channel == "ch01":
                    return "barcode"
            elif "mask" in cycle:
                if channel == "ch00":
                    return "fiducial"
                elif channel == "ch01":
                    return "primer"
        elif "_block3D" in filename or "shifts" in filename:
            return "fiducial"

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

    def get_path_from_type(self, data_type):
        # TODO: Implement this method
        pass

    def get_cycle_from_path(self, data_path):
        split_path = os.path.basename(data_path).split("_")
        if len(split_path) >= 3:
            return split_path[2]
        else:
            return None

    def get_channel_from_path(self, data_path):
        split_path = os.path.basename(data_path).split("_")
        if len(split_path) >= 8:
            return split_path[7]
        else:
            return None

    def get_path_from_cycle(self, cycle, data_paths):
        for data_path in data_paths:
            if cycle in data_path:
                return data_path
        raise ValueError(
            f"No data found for cycle {cycle} inside data_paths: \n{data_paths}"
        )

    def dict_elt_have_same_length(self, dict):
        if len(set([len(v) for v in dict.values()])) != 1:
            raise ValueError(
                "All values must have the same length in the dictionary:\n", dict
            )

    def get_analysis_types(self):
        # TODO: Implement this method
        pass

    def get_sup_paths_by_cycle(self, sup_types_to_find):
        """
        Retrieves supplementary data paths by cycle for the given supplementary data types.

        Args:
            sup_types_to_find (list): A list of supplementary data types to find.
                                        List[DataType]

        Returns:
            dict: A dictionary containing path of supplementary data to find grouped by cycle and type.
                  The keys are cycle names and the values are dictionaries where the keys are
                  supplementary types and the values are the corresponding path.
                  Dict[CycleName, Dict[DataType, Path]]
        """
        supplementary_paths_by_type = {
            sup_type: self.get_path_from_type(sup_type)
            for sup_type in sup_types_to_find
        }
        self.dict_elt_have_same_length(supplementary_paths_by_type)
        sup_paths_by_cycle = {}
        for type, paths in supplementary_paths_by_type.items():
            for path in paths:
                cycle = self.get_cycle_from_path(path)
                if cycle not in sup_paths_by_cycle:
                    sup_paths_by_cycle[cycle] = {}
                sup_paths_by_cycle[cycle][type] = path
        return sup_paths_by_cycle
