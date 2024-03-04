import os
import json
from typing import List
from core_types import get_data_type, first_type_accept_second, DataType
from core_types import AnalysisType as at


def get_file_path(directory, filename, extension):
    return os.path.join(directory, f"{filename}.{extension}")


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


def load_json(file_path):
    with open(file_path, "r") as file:
        print(f"[Loading] {file_path}")
        return json.load(file)


def save_ecsv(table, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    table.write(path, format="ascii.ecsv", overwrite=True)


class DataManager:
    def __init__(self, run_args):
        self.input_folder = run_args.input
        self.output_folder = run_args.output
        self.input_files = extract_files(self.input_folder)
        self.parameters_file = self.get_parameters_file()
        self.parameters = load_json(self.parameters_file)
        self.analysis_files = self.get_analysis_files()

    def get_parameters_file(self):
        for file in self.input_files:
            if file[1] in ["parameters", "infoList"] and file[2] == "json":
                return file[0]
        raise FileNotFoundError(
            "No parameters file found in input folder: ", self.input_folder
        )

    def get_analysis_files(self):
        analysis_files = {
            at.FIDUCIAL: [],
            at.BARCODE: [],
            at.DAPI: [],
            at.RNA: [],
            at.PRIMER: [],
            at.TRACE: [],
        }
        for file in self.input_files:
            analysis_type = self.get_analysis_type(file[1], file[2])
            data_type = get_data_type(file[1], file[2])
            if analysis_type and data_type:
                analysis_files[analysis_type].append((data_type, file[0]))
                # Manage the specific case of the TRACE type
                if first_type_accept_second(DataType.SEGMENTED, data_type):
                    analysis_files[at.TRACE].append((data_type, file[0]))
        return analysis_files

    def _get_paths_from_analysis_and_data_type(self, analysis_type, data_type):
        return [
            path
            for type, path in self.analysis_files[analysis_type]
            if first_type_accept_second(data_type, type)
        ]

    def get_paths_from_analysis_and_data_type(self, analysis_type, data_type):
        if isinstance(data_type, list):
            paths = self._get_paths_from_analysis_and_data_type(
                analysis_type, data_type[0]
            )
            if not paths:
                return self._get_paths_from_analysis_and_data_type(
                    analysis_type, data_type[1]
                )
            return paths
        return self._get_paths_from_analysis_and_data_type(analysis_type, data_type)

    def get_analysis_type(self, filename, extension):
        if extension in ["png", "log", "md"] or filename in ["parameters", "infoList"]:
            return None
        elif extension in ["tif", "tiff", "npy"]:
            cycle = self.get_cycle_from_path(filename)
            channel = self.get_channel_from_path(
                filename
            )  # TODO check ch depending parameters
            if cycle == "DAPI":
                if channel == "ch00":
                    return at.DAPI
                elif channel == "ch01":
                    return at.FIDUCIAL
                elif channel == "ch02":
                    return at.RNA
            elif "RT" in cycle:
                if channel == "ch00":
                    return at.FIDUCIAL
                elif channel == "ch01":
                    return at.BARCODE
            elif "mask" in cycle:
                if channel == "ch00":
                    return at.FIDUCIAL
                elif channel == "ch01":
                    return at.PRIMER
            elif "matrix" in filename:
                return at.TRACE
        elif "_block3D" in filename or filename in ["shifts", "register_global"]:
            return at.FIDUCIAL
        elif extension in ["dat", "ecsv"]:
            if "Trace" in filename or "_barcode" in filename:
                return at.TRACE
        else:
            raise ValueError(
                f"File {filename}.{extension} does not match any analysis type."
            )

    def get_paths_from_type(self, data_type, analysis_type):
        print(f"Looking for {data_type} in {analysis_type}")
        return [
            path
            for type, path in self.analysis_files[analysis_type]
            if first_type_accept_second(data_type, type)
        ]

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
        if dict and len(set([len(v) for v in dict.values()])) != 1:
            raise ValueError(
                "All values must have the same length in the dictionary:\n", dict
            )

    def get_analysis_types(self):
        order = ["fiducial", "barcode", "DAPI", "RNA", "primer", "trace"]
        order = [at(x) for x in order]
        analysis_types = [x for x in order if len(self.analysis_files[x]) > 0]
        return analysis_types

    def get_sup_paths_by_cycle(self, sup_types_to_find, analysis_type):
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
            sup_type: self.get_paths_from_type(sup_type, analysis_type)
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
