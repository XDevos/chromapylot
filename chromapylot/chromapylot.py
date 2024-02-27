from typing import List, Literal, Dict
from enum import Enum
from .data_manager import DataManager
from .run_args import RunArgs
import os
import module as mod

DataType = Literal[
    "_3d",
    "_2d",
    "shift_tuple",
    "shift_dict",
    "shift_table",
    "_skipped",
    "_shifted",
    "_segmented",
    "_selected",
    "_table",
    "_filtered",
    "_registered",
]


class DataType(Enum):
    IMAGE_3D = "IMAGE_3D"
    IMAGE_2D = "IMAGE_2D"
    SHIFT_TUPLE = "SHIFT_TUPLE"
    SHIFT_DICT = "SHIFT_DICT"
    REGISTRATION_TABLE = "REGISTRATION_TABLE"
    IMAGE_3D_SHIFTED = "IMAGE_3D_SHIFTED"
    IMAGE_3D_SEGMENTED = "IMAGE_3D_SEGMENTED"
    IMAGE_2D_SHIFTED = "IMAGE_2D_SHIFTED"
    IMAGE_2D_SEGMENTED = "IMAGE_2D_SEGMENTED"
    TABLE_3D = "TABLE_3D"
    TABLE_2D = "TABLE_2D"
    TABLE = "TABLE"
    TABLE_FILTERED = "TABLE_FILTERED"
    TABLE_3D_REGISTERED = "TABLE_3D_REGISTERED"
    TRACE_TABLE_3D = "TRACE_TABLE_3D"
    TRACE_TABLE_2D = "TRACE_TABLE_2D"
    MATRIX_3D = "MATRIX_3D"
    MATRIX_2D = "MATRIX_2D"


class AnalysisType(Enum):
    FIDUCIAL = "fiducial"
    BARCODE = "barcode"
    DAPI = "dapi"
    RNA = "rna"
    PRIMER = "primer"
    SATELLITE = "satellite"
    TRACE = "trace"


ModuleName = Literal[
    "project",
    "register_global",
    "skip",
    "shift",
    "shift_fiducial",
    "register_local",
    "filter_table",
    "filter_mask",
    "segment",
    "extract",
    "filter_localization",
    "register_localization",
    "build_trace",
    "build_matrix",
]

ExampleType = Literal[
    "fiducial_3d",
    "fiducial_2d",
    "shift_tuple",
    "shift_dict",
    "shift_table",
    "fiducial_3d_shifted",
    "barcode_3d",
    "barcode_3d_shifted",
    "barcode_3d_segmented",
    "barcode_3d_table",
    "barcode_3d_table_filtered",
    "barcode_3d_table_registered",
    "barcode_3d_table_filtered_registered",
    "barcode_3d_table_registered_filtered",
    "mask_3d",
    "mask_3d_shifted",
    "mask_3d_segmented",
    "mask_3d_table",
    "mask_3d_table_filtered",
]


class Pipeline:
    def __init__(self, modules: List[mod.Module]):
        self.modules = modules
        self.supplementary_data = {}

    def prepare(self):
        first_input_type = self.modules[0].input_type
        generated_data_type = [first_input_type]
        supplementary_data_to_find = []
        for module in self.modules:
            # Load data to keep during all processes
            module.load_reference_data()
            # Collect data type to keep during one process
            if module.supplementary_data:
                self.supplementary_data.update(module.supplementary_data)
                if module.supplementary_type not in generated_data_type:
                    supplementary_data_to_find.append(module.supplementary_type)
            generated_data_type.append(module.output_type)
        return first_input_type, supplementary_data_to_find

    def choose_to_keep_data(self, module, data):
        if module.output_type in self.supplementary_data:
            self.supplementary_data[module.output_type] = data

    def choose_to_keep_input_data(self, data):
        if self.modules[0].input_type in self.supplementary_data:
            self.supplementary_data[self.modules[0].input_type] = data

    def update_supplementary_data(self, supplementary_paths):
        for key, value in supplementary_paths.items():
            if key not in self.supplementary_data:
                raise ValueError(f"Supplementary data type {key} not found.")
            elif self.supplementary_data[key] is not None:
                raise ValueError(f"Supplementary data type {key} already exists.")
            else:
                self.supplementary_data[key] = value

    def load_supplementary_data(self, module: mod.Module, cycle: str):
        data_type = module.supplementary_type
        if data_type:
            if data_type in self.supplementary_data:
                if self.supplementary_data[data_type] is None:
                    self.supplementary_data[data_type] = module.load_supplementary_data(
                        os.getcwd(), cycle
                    )
                elif isinstance(self.supplementary_data[data_type], str):
                    self.supplementary_data[data_type] = module.load_supplementary_data(
                        self.supplementary_data[data_type], cycle
                    )
                return self.supplementary_data[data_type]
            else:
                raise ValueError(f"Supplementary data {data_type} not found.")
        else:
            return None

    def process(
        self, data_path: str, supplementary_paths: Dict[DataType, str], cycle: str
    ):
        print(f"Processing data from cycle {cycle}.")
        data = self.modules[0].load_data(data_path)
        self.choose_to_keep_input_data(data)
        self.update_supplementary_data(supplementary_paths)
        for module in self.modules:
            supplementary_data = self.load_supplementary_data(module, cycle)
            if module.switched:
                data, supplementary_data = supplementary_data, data
            data = module.run(data, supplementary_data)
            module.save_data(data_path, data)
            self.choose_to_keep_data(module, data)


class AnalysisManager:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.ordered_analysis_types = self.sort_analysis_types()
        self.module_names = []
        self.pipelines: Dict[AnalysisType, Pipeline] = {
            "fiducial": None,
            "barcode": None,
            "dapi": None,
            "rna": None,
            "primer": None,
            "satellite": None,
            "trace": None,
        }

    def parse_commands(self, commands: List[str]):
        self.module_names = list(set(commands))

    def get_module_chain(self, pipeline_type: AnalysisType) -> List[ModuleName]:
        if pipeline_type == "fiducial":
            return ["project", "register_global", "shift", "register_local"]
        elif pipeline_type == "barcode":
            return ["shift", "segment", "extract"]
        elif (
            pipeline_type == "dapi"
            or pipeline_type == "rna"
            or pipeline_type == "primer"
            or pipeline_type == "satellite"
        ):
            return ["shift", "segment", "extract", "filter_table", "filter_mask"]
        elif pipeline_type == "trace":
            return [
                "filter_localization",
                "register_localization",
                "build_trace",
                "build_matrix",
            ]

    def create_module(self, module_name: ModuleName, pipeline_type: AnalysisType):
        module_params = self.data_manager.get_module_params(module_name, pipeline_type)
        module_mapping = {
            "project": mod.ProjectModule,
            "skip": mod.SkipModule,
            "shift_2d": mod.Shift2DModule,
            "shift_3d": mod.Shift3DModule,
            "register_global": mod.RegisterGlobalModule,
            "register_local": mod.RegisterLocalModule,
            "segment_2d": mod.Segment2DModule,
            "segment_3d": mod.Segment3DModule,
            "extract_2d": mod.Extract2DModule,
            "extract_3d": mod.Extract3DModule,
            "filter_mask": mod.FilterMaskModule,
            "select_mask_2d": mod.SelectMask2DModule,
            "select_mask_3d": mod.SelectMask3DModule,
            "filter_localization": mod.FilterLocalizationModule,
            "register_localization": mod.RegisterLocalizationModule,
            "build_trace": mod.BuildTraceModule,
            "build_matrix": mod.BuildMatrixModule,
        }
        if module_name in module_mapping:
            return module_mapping[module_name](module_params)
        else:
            raise ValueError(f"Module {module_name} does not exist.")

    def create_pipeline_modules(self, pipeline_type: AnalysisType):
        module_chain = self.get_module_chain(pipeline_type)
        modules = []
        for i in range(len(module_chain)):
            if module_chain[i] in self.module_names:
                modules.append(self.create_module(module_chain[i], pipeline_type))
                # check if we don't break the chain
                if (
                    len(modules) >= 2
                    and modules[-2].output_type != modules[-1].input_type
                ):
                    if modules[-2].output_type == modules[-1].supplementary_type:
                        modules[-1].switch_input_supplementary()
                    else:
                        raise ValueError(
                            f"Module {module_chain[i]} cannot be used without {module_chain[i - 1]}, for {pipeline_type} analysis."
                        )

    def sort_analysis_types(self):
        analysis_types = self.data_manager.get_analysis_types()
        order = ["fiducial", "barcode", "dapi", "rna", "primer", "satellite", "trace"]
        analysis_types = [x for x in order if x in analysis_types]
        return analysis_types

    def create_pipelines(self):
        for analysis_type in self.ordered_analysis_types:
            modules = self.create_pipeline_modules(analysis_type)
            self.pipelines[analysis_type] = Pipeline(modules)

    def launch_analysis(self):
        for analysis_type in self.ordered_analysis_types:
            input_type, sup_types_to_find = self.pipelines[analysis_type].prepare(
                self.data_manager
            )
            input_paths = self.data_manager.get_path_from_type(input_type)
            sup_paths = self.data_manager.get_sup_paths_by_cycle(sup_types_to_find)
            for data_path in input_paths:
                cycle = self.data_manager.get_cycle_from_path(data_path)
                if cycle not in sup_paths:
                    sup_paths[cycle] = {}
                self.pipelines[analysis_type].process(
                    data_path, sup_paths.pop(cycle), cycle
                )
            if sup_paths:
                raise ValueError(
                    f"Supplementary data not used for analysis {analysis_type}: {sup_paths}"
                )


if __name__ == "__main__":
    run_args = RunArgs()
    data_manager = DataManager(run_args)
    analysis_manager = AnalysisManager(data_manager)
    analysis_manager.parse_commands(run_args.commands)
    analysis_manager.create_pipelines()
    analysis_manager.launch_analysis()

