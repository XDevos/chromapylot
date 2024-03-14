#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Dict, Union, Any
from data_manager import DataManager
from run_args import RunArgs
import modules.module as mod
from datetime import datetime
from core_types import DataType, AnalysisType, CommandName
from parameters import (
    ProjectionParams,
    RegistrationParams,
    SegmentationParams,
    MatrixParams,
    PipelineParams,
)
from modules.build_trace import BuildTrace3DModule
from modules.project import ProjectModule


class Pipeline:
    def __init__(self, analysis_type: AnalysisType, modules: List[mod.Module]):
        self.analysis_type = analysis_type
        self.modules = modules
        self.supplementary_data: Dict[DataType, Any] = {}

    def prepare(self, data_manager: DataManager):
        first_input_type = self.modules[0].input_type
        generated_data_type: List[DataType] = [first_input_type]
        supplementary_data_to_find = []
        for module in self.modules:
            if module.reference_type is not None:
                # Load data to keep during all processes
                print(
                    f"Loading reference data for {module.__class__.__name__} with {self.analysis_type} and {module.reference_type}."
                )
                paths = data_manager.get_paths_from_analysis_and_data_type(
                    self.analysis_type, module.reference_type
                )
                module.load_reference_data(paths)
            # Collect data type to keep during one process
            if module.supplementary_type is not None:
                if isinstance(module.supplementary_type, list):
                    supp_data_found = False
                    for i in range(len(module.supplementary_type)):
                        if (
                            not supp_data_found
                            and module.supplementary_type[i] in generated_data_type
                        ):
                            supp_data_found = True
                            self.supplementary_data[module.supplementary_type[i]] = None
                            print(
                                f"Replace {module.supplementary_type} by {module.supplementary_type[i]}."
                            )
                            module.supplementary_type = module.supplementary_type[i]
                    if not supp_data_found:
                        supplementary_data_to_find.append(module.supplementary_type)
                else:
                    self.supplementary_data[module.supplementary_type] = None
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
                        None, cycle
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
        self,
        data_path: str,
        output_dir: str,
        supplementary_paths: Dict[DataType, str],
        cycle: str,
    ):
        print(f"Processing data from cycle {cycle}.")
        data = self.modules[0].load_data(data_path)
        self.choose_to_keep_input_data(data)
        self.update_supplementary_data(supplementary_paths)
        for module in self.modules:
            supplementary_data = self.load_supplementary_data(module, cycle)
            if module.switched:
                data, supplementary_data = supplementary_data, data
            if supplementary_data is None:
                data = module.run(data)
            else:
                data = module.run(data, supplementary_data)
            module.save_data(data, output_dir, data_path)
            self.choose_to_keep_data(module, data)


class AnalysisManager:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.found_analysis_types = self.data_manager.get_analysis_types()
        self.analysis_to_process = []
        self.module_names = []
        self.pipelines: Dict[AnalysisType, Pipeline] = {
            "fiducial": {2: None, 3: None},
            "barcode": {2: None, 3: None},
            "DAPI": {2: None, 3: None},
            "RNA": {2: None, 3: None},
            "primer": {2: None, 3: None},
            "trace": {2: None, 3: None},
        }

    def parse_commands(self, commands: List[str]):
        self.module_names = list(set(commands))

    def get_module_chain(
        self, pipeline_type: AnalysisType, dim: int
    ) -> List[CommandName]:
        if pipeline_type == AnalysisType.FIDUCIAL:
            # TODO: WARNING for fiducial analysis type, if dim = 23, just execute the 3D pipeline
            if dim == 2:
                return ["skip", "project", "register_global"]
            elif dim == 3:
                return [
                    "skip",
                    "project",
                    "register_global",
                    "shift_3d",
                    "register_local",
                ]
        elif pipeline_type == AnalysisType.BARCODE:
            if dim == 2:
                return ["skip", "project", "shift_2d", "segment_2d", "extract_2d"]
            elif dim == 3:
                return ["skip", "shift_3d", "segment_3d", "extract_3d"]
        elif (
            pipeline_type == AnalysisType.DAPI
            or pipeline_type == AnalysisType.RNA
            or pipeline_type == AnalysisType.PRIMER
        ):
            if dim == 2:
                return [
                    "skip",
                    "project",
                    "shift_2d",
                    "segment_2d",
                    "extract_2d",
                    "filter_mask",
                    "select_mask_2d",
                ]
            elif dim == 3:
                return [
                    "skip",
                    "shift_3d",
                    "segment_3d",
                    "extract_3d",
                    "filter_mask",
                    "select_mask_3d",
                ]
        elif pipeline_type == AnalysisType.TRACE:
            if dim == 2:
                return [
                    "filter_localization",
                    "build_trace_2d",
                    "build_matrix_2d",
                ]
            elif dim == 3:
                return [
                    "filter_localization",
                    "register_localization",
                    "build_trace_3d",
                    "build_matrix_3d",
                ]
        else:
            raise ValueError(
                f"Analysis type '{pipeline_type}' with dimension '{dim}' not found."
            )

    def create_module(
        self,
        module_name: CommandName,
        module_params: Dict[
            str,
            Union[
                ProjectionParams, RegistrationParams, SegmentationParams, MatrixParams
            ],
        ],
    ):
        module_mapping = {
            "project": ProjectModule,
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
            "build_trace_3d": BuildTrace3DModule,
            "build_matrix": mod.BuildMatrixModule,
        }
        if module_name in module_mapping:
            print(f"Creating module {module_name}.")
            return module_mapping[module_name](**module_params)
        else:
            raise ValueError(f"Module {module_name} does not exist.")

    def create_pipeline_modules(self, pipeline_type: AnalysisType, dim: int):
        module_chain = self.get_module_chain(pipeline_type, dim)
        modules: List[mod.Module] = []
        pipe_params = PipelineParams(self.data_manager.parameters, pipeline_type)
        for i in range(len(module_chain)):
            if module_chain[i] in self.module_names:
                print(
                    f"[Pipeline-{pipeline_type.name}] Checking module {module_chain[i]}."
                )
                module_params = pipe_params.get_module_params(module_chain[i])
                modules.append(self.create_module(module_chain[i], module_params))
                # check if we don't break the chain
                if len(modules) >= 2 and not modules[-1].is_compatible(
                    modules[-2].output_type
                ):
                    raise ValueError(
                        f"Module {module_chain[i]} cannot be used without {module_chain[i - 1]}, for {pipeline_type} analysis."
                    )
        return modules

    def create_pipelines(self, dims: List[int] = [2, 3]):
        for analysis_type in self.found_analysis_types:
            for dim in dims:
                modules = self.create_pipeline_modules(analysis_type, dim)
                if modules:
                    self.analysis_to_process.append((analysis_type, dim))
                    print(f"Creating pipeline for analysis {analysis_type}")
                    self.pipelines[analysis_type.value][dim] = Pipeline(
                        analysis_type, modules
                    )
                else:
                    print(
                        f"No pipeline to create for analysis {analysis_type} with dimension {dim}."
                    )
        self.check_fiducial_dimension(dims)
        if not self.analysis_to_process:
            raise ValueError("No analysis to process.")

    def check_fiducial_dimension(self, dims: List[int]):
        pass
        # if len(dims) == 2 and AnalysisType.FIDUCIAL in self.analysis_to_process:
        #     pipe = self.pipelines["fiducial"][3]
        # self.pipe.modules.remove("skip")
        # self.pipe.modules.remove("project")
        # self.pipe.modules.remove("register_global")

    def launch_analysis(self):
        output_dir = self.data_manager.output_folder
        for analysis_type, dim in self.analysis_to_process:
            print(f"Launching analysis {analysis_type}")
            pipe = self.pipelines[analysis_type.value][dim]
            input_type, sup_types_to_find = pipe.prepare(self.data_manager)
            input_paths = self.data_manager.get_paths_from_type(
                input_type, analysis_type
            )
            sup_paths = self.data_manager.get_sup_paths_by_cycle(
                sup_types_to_find, analysis_type
            )
            for data_path in input_paths:
                cycle = self.data_manager.get_cycle_from_path(data_path)
                if cycle not in sup_paths:
                    sup_paths[cycle] = {}
                pipe.process(data_path, output_dir, sup_paths.pop(cycle), cycle)
            if sup_paths:
                raise ValueError(
                    f"Supplementary data not used for analysis {analysis_type}: {sup_paths}"
                )


def main(command_line_args=None):
    begin_time = datetime.now()
    run_args = RunArgs(command_line_args)
    data_manager = DataManager(run_args)
    analysis_manager = AnalysisManager(data_manager)
    analysis_manager.parse_commands(run_args.commands)
    analysis_manager.create_pipelines(run_args.dimension)
    analysis_manager.launch_analysis()
    print("\n==================== Normal termination ====================\n")
    print(f"Elapsed time: {datetime.now() - begin_time}")


if __name__ == "__main__":
    main()
