from typing import List, Dict, Union
from enum import Enum
from .data_manager import DataManager
from .run_args import RunArgs
import os
from . import module as mod
from .types import DataType, AnalysisType, ModuleName
from .parameters import ProjectionParams, RegistrationParams, SegmentationParams, MatrixParams, PipelineParams

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
            "trace": None,
        }

    def parse_commands(self, commands: List[str]):
        self.module_names = list(set(commands))

    def get_module_chain(self, pipeline_type: AnalysisType) -> List[ModuleName]:
        if pipeline_type == AnalysisType.FIDUCIAL:
            return ["skip", "project", "register_global", "shift_3d", "register_local"]
        elif pipeline_type == AnalysisType.BARCODE:
            return ["shift_3d", "segment", "extract"]
        elif (
            pipeline_type == AnalysisType.DAPI
            or pipeline_type == AnalysisType.RNA
            or pipeline_type == AnalysisType.PRIMER
        ):
            return ["shift_3d", "segment", "extract", "filter_table", "filter_mask"]
        elif pipeline_type == AnalysisType.TRACE:
            return [
                "filter_localization",
                "register_localization",
                "build_trace",
                "build_matrix",
            ]

    def create_module(self, module_name: ModuleName, module_params: Union[ProjectionParams, RegistrationParams, SegmentationParams, MatrixParams]):
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
            print(f"Creating module {module_name} with parameters {module_params}.")
            return module_mapping[module_name](module_params)
        else:
            raise ValueError(f"Module {module_name} does not exist.")

    def create_pipeline_modules(self, pipeline_type: AnalysisType):
        module_chain = self.get_module_chain(pipeline_type)
        modules = []
        pipe_params = PipelineParams(self.data_manager.parameters, pipeline_type)
        for i in range(len(module_chain)):
            if module_chain[i] in self.module_names:
                module_params = pipe_params.get_module_params(module_chain[i])
                modules.append(self.create_module(module_chain[i], module_params))
                # check if we don't break the chain
                if (
                    len(modules) >= 2
                    and modules[-2].output_type != modules[-1].input_type
                ):
                    print(f"modules[-2].output_type = {modules[-2].output_type}")
                    print(f"modules[-1].input_type = {modules[-1].input_type}")
                    print(f"modules[-1].supplementary_type = {modules[-1].supplementary_type}")
                    if modules[-2].output_type == modules[-1].supplementary_type:
                        modules[-1].switch_input_supplementary()
                    else:
                        print(f"modules: {modules}")
                        raise ValueError(
                            f"Module {module_chain[i]} cannot be used without {module_chain[i - 1]}, for {pipeline_type} analysis."
                        )

    def sort_analysis_types(self):
        analysis_types = self.data_manager.get_analysis_types()
        print(f"Analysis types found: {analysis_types}")
        order = ["fiducial", "barcode", "DAPI", "RNA", "primer", "trace"]
        order = [AnalysisType(x) for x in order]
        analysis_types = [x for x in order if x in analysis_types]
        print(f"Analysis types to run: {analysis_types}")
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
            input_paths = self.data_manager.get_paths_from_type(input_type, analysis_type)
            sup_paths = self.data_manager.get_sup_paths_by_cycle(sup_types_to_find, analysis_type)
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

def main():
    run_args = RunArgs()
    data_manager = DataManager(run_args)
    analysis_manager = AnalysisManager(data_manager)
    analysis_manager.parse_commands(run_args.commands)
    analysis_manager.create_pipelines()
    analysis_manager.launch_analysis()


if __name__ == "__main__":
    main()