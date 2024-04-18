import json
import os
from typing import List

from skimage import io
import numpy as np

from chromapylot.core.core_types import AnalysisType as at
from chromapylot.core.core_types import (
    DataType,
    first_type_accept_second,
    get_data_type,
)

from dask.distributed import Lock


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
    return files


def load_json(file_path):
    with open(file_path, "r") as file:
        print(f"[Load] Parameters")
        print(f"> {file_path}")
        return json.load(file)


def save_json(data, path):
    try:
        with Lock(path):
            with open(path, "w") as file:
                json.dump(data, file, ensure_ascii=False, sort_keys=True, indent=4)
    # Case where we don't have a dask client
    except RuntimeError:
        with open(path, "w") as file:
            json.dump(data, file, ensure_ascii=False, sort_keys=True, indent=4)


def save_ecsv(table, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    table.write(path, format="ascii.ecsv", overwrite=True)
    print(f"[Saving] {path}")


def save_npy(array, path, out_dir_length):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if array.shape <= (1, 1):
        raise ValueError(f"Image is empty! Expected file {path} to be saved.")
    np.save(path, array)

    short_path = path[out_dir_length:]
    print(f"> $OUTPUT{short_path}")


def get_roi_number_from_image_path(image_path):
    return os.path.basename(image_path).split("_")[3]


class DataManager:
    def __init__(self, run_args):
        self.input_folder = run_args.input
        self.in_dir_len = len(self.input_folder)
        self.output_folder = run_args.output
        self.out_dir_len = len(self.output_folder)
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

    def refresh_input_files(self):
        self.input_files = extract_files(self.input_folder)
        self.analysis_files = self.get_analysis_files()

    def get_analysis_files(self):
        analysis_files = {
            at.REFERENCE: [],
            at.FIDUCIAL: [],
            at.BARCODE: [],
            at.DAPI: [],
            at.RNA: [],
            at.PRIMER: [],
            at.TRACE: [],
        }
        ref_cycle = self.get_ref_cycle()
        for file in self.input_files:
            analysis_types = self.get_analysis_type(file[1], file[2])
            data_type = get_data_type(file[1], file[2])
            for analysis in analysis_types:
                if analysis and data_type:
                    analysis_files[analysis].append((data_type, file[0]))
                    # Manage the specific case of the REFERENCE type
                    if ref_cycle and analysis == at.FIDUCIAL and ref_cycle in file[1]:
                        analysis_files[at.REFERENCE].append(
                            analysis_files[analysis].pop()
                        )
                    # Manage the specific case of the TRACE type
                    # TODO: see if we can refacto with the update of the analysis_types (became a list of AnalysisType)
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
        if analysis_type == at.FIDUCIAL:
            analysis_type = at.REFERENCE
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
        if extension in ["png", "log", "md", "table", "py", None] or filename in [
            "parameters",
            "parameters_loaded",
            "infoList",
        ]:
            return []
        elif extension in ["tif", "tiff", "npy"]:
            cycle = self.get_cycle_from_path(filename)
            channel = self.get_channel_from_path(
                filename
            )  # TODO check ch depending parameters
            if cycle == "DAPI":
                if channel == "ch00":
                    analysis_type = [at.DAPI]
                elif channel == "ch01":
                    analysis_type = [at.FIDUCIAL]
                elif channel == "ch02":
                    analysis_type = [at.RNA]
            elif "RT" in cycle:
                if channel == "ch00":
                    analysis_type = [at.FIDUCIAL]
                elif channel == "ch01":
                    analysis_type = [at.BARCODE]
            elif "mask" in cycle:
                if channel == "ch00":
                    analysis_type = [at.FIDUCIAL]
                elif channel == "ch01":
                    analysis_type = [at.PRIMER]
            elif "matrix" in filename:
                analysis_type = [at.TRACE]
        elif "_block3D" in filename or filename in [
            "shifts",
            "register_global",
            "alignImages",
        ]:
            # affect list of all analysis types
            analysis_type = list(at)
        elif extension in ["dat", "ecsv"]:
            if "Trace" in filename or "_barcode" in filename:
                analysis_type = [at.TRACE]
        else:
            raise ValueError(
                f"File {filename}.{extension} does not match any analysis type."
            )
        return analysis_type

    def get_ref_cycle(self):
        return (
            self.parameters.get("common", {})
            .get("alignImages", {})
            .get("referenceFiducial")
        )

    def get_paths_from_type(self, data_type, analysis_type):
        paths = [
            path
            for type, path in self.analysis_files[analysis_type]
            if first_type_accept_second(data_type, type)
        ]
        print(f"[Found] {len(paths)} {data_type.value} to process")
        return paths

    @staticmethod
    def get_cycle_from_path(data_path):
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
        order = ["reference", "fiducial", "barcode", "DAPI", "RNA", "primer", "trace"]
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

    def load_image_3d(self, path):
        print(f"[Load] IMAGE_3D")
        short_path = path[self.in_dir_len :]
        print(f"> $INPUT{short_path}")
        return io.imread(path).squeeze()

    def get_3d_ref_filename(self):
        for data_type, filepath in self.analysis_files[at.REFERENCE]:
            if data_type == DataType.IMAGE_3D:
                return os.path.basename(filepath)


def tif_path_to_projected(tif_path):
    base = os.path.basename(tif_path).split(".")[0]
    directory = os.path.dirname(tif_path)
    projected_path = os.path.join(directory, "project", "data", f"{base}_2d.npy")
    return projected_path


def create_dirs_from_path(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))


def get_init_basename(filepath):
    # Remove extension
    base = os.path.basename(filepath).split(".")[0]
    # Remove _2d suffix if present. TODO: generalize this to all suffixes
    base = base[:-3] if base[-3:] == "_2d" else base
    return base


def create_png_path(init_filename, out_dir, module_dir, postfix):
    base = get_init_basename(init_filename)
    png_filename = base + postfix + ".png"
    out_path = os.path.join(out_dir, module_dir, png_filename)
    create_dirs_from_path(out_path)
    return out_path


def create_npy_path(init_filename, out_dir, module_dir, postfix):
    base = get_init_basename(init_filename)
    npy_filename = base + postfix + ".npy"
    out_path = os.path.join(out_dir, module_dir, "data", npy_filename)
    create_dirs_from_path(out_path)
    return out_path
