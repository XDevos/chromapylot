from typing import Any, Dict, List, Union
from dask import delayed, compute
from dask.distributed import Client

from chromapylot.routines import routine
from routines.build_trace import BuildTrace3D
from routines.project import (
    ProjectModule,
    ProjectByBlockModule,
    InterpolateFocalPlane,
    SplitInBlocks,
)
from routines.register_global import (
    RegisterGlobalModule,
    RegisterByBlock,
    CompareBlockGlobal,
)
from routines.register_local import RegisterLocal, Preprocess3D
from chromapylot.core.core_types import AnalysisType, RoutineName
from chromapylot.core.data_manager import DataManager
from chromapylot.parameters.matrix_params import MatrixParams
from chromapylot.parameters.params_manager import ParamsManager
from chromapylot.parameters.projection_params import ProjectionParams
from chromapylot.parameters.registration_params import RegistrationParams
from chromapylot.parameters.segmentation_params import SegmentationParams
from chromapylot.routines.localize import (
    Localize2D,
    ExtractProperties,
    ReducePlanes,
    FitSubpixel,
    ShiftSpotOnZ,
    AddCycleToTable,
)
from chromapylot.routines.segment import Segment2D, Segment3D, Deblend3D


from chromapylot.core.pipeline import Pipeline

from chromapylot.core.core_logging import print_analysis_type

from chromapylot.core.core_logging import print_text_inside
import multiprocessing
import os

import numpy as np
from dask.distributed import Client, LocalCluster


class AnalysisManager:
    def __init__(
        self,
        data_manager: DataManager,
        dims: List[int] = [2, 3],
        n_threads: int = 1,
        analysis_types: List[AnalysisType] = [],
    ):
        self.data_manager = data_manager
        self.dims = dims
        self.n_threads = n_threads
        self.user_analysis_types = analysis_types
        self.found_analysis_types = self.data_manager.get_analysis_types()
        self.analysis_to_process = []
        self.routine_names = []
        self.pipelines: Dict[AnalysisType, Dict[int, Pipeline]] = {
            "reference": {2: None, 3: None},
            "fiducial": {2: None, 3: None},
            "barcode": {2: None, 3: None},
            "DAPI": {2: None, 3: None},
            "RNA": {2: None, 3: None},
            "primer": {2: None, 3: None},
            "trace": {2: None, 3: None},
        }

    def parse_commands(self, commands: List[RoutineName]):
        self.routine_names = list(set(commands))

    def get_routine_chain(
        self, pipeline_type: AnalysisType, dim: int
    ) -> List[RoutineName]:
        if pipeline_type == AnalysisType.REFERENCE:
            if dim == 2:
                chain = ["project"]
                if len(self.dims) == 2:
                    print(
                        "> If both dimensions are required, there isn't pipeline for REFERENCE 2D."
                    )
                    chain = []
            elif dim == 3:
                chain = ["project"]
        elif pipeline_type == AnalysisType.FIDUCIAL:
            if dim == 2:
                chain = ["skip", "project", "register_global"]
                if len(self.dims) == 2:
                    # WARNING for fiducial analysis type, if dim = 23, just execute the 3D pipeline
                    print(
                        "> If both dimensions are required, there isn't pipeline for FIDUCIAL 2D."
                    )
                    chain = []
            elif dim == 3:
                chain = [
                    "skip",
                    "preprocess_3d",
                    "project",
                    "register_global",
                    "shift_3d",
                    "register_local",
                ]
        elif pipeline_type == AnalysisType.BARCODE:
            if dim == 2:
                chain = ["skip", "project", "shift_2d", "localize_2d"]
            elif dim == 3:
                chain = [
                    "skip",
                    "reduce_planes",
                    "preprocess_3d",
                    "shift_3d",
                    "segment_3d",
                    "deblend_3d",
                    "extract_properties",
                    "add_cycle_to_table",
                    "fit_subpixel",
                    "shift_spot_on_z",
                ]
        elif (
            pipeline_type == AnalysisType.DAPI
            or pipeline_type == AnalysisType.RNA
            or pipeline_type == AnalysisType.PRIMER
        ):
            if dim == 2:
                chain = [
                    "skip",
                    "project",
                    "shift_2d",
                    "segment_2d",
                    "extract_2d",
                    "filter_mask",
                    "select_mask_2d",
                ]
            elif dim == 3:
                chain = [
                    "skip",
                    "shift_3d",
                    "segment_3d",
                    "extract_properties",
                    "filter_mask",
                    "select_mask_3d",
                ]
        elif pipeline_type == AnalysisType.TRACE:
            if dim == 2:
                chain = [
                    "filter_localization",
                    "build_trace_2d",
                    "build_matrix",
                ]
            elif dim == 3:
                chain = [
                    "filter_localization",
                    "register_localization",
                    "build_trace_3d",
                    "build_matrix",
                ]
        else:
            raise ValueError(
                f"Analysis type '{pipeline_type}' with dimension '{dim}' not found."
            )
        chain = self.explicite_intern_routine(chain, pipeline_type)
        chain = self.convert_string_to_routine_name(chain)
        return chain

    def convert_string_to_routine_name(self, chain: List[str]) -> List[RoutineName]:
        return [RoutineName(routine) for routine in chain]

    def explicite_intern_routine(
        self, chain: List[RoutineName], pipeline_type
    ) -> List[RoutineName]:
        # Projection Laplacian
        try:
            index_chain = chain.index("project")
            pipe_params = ParamsManager(self.data_manager.parameters, pipeline_type)
            if pipe_params.projection.mode == "laplacian":
                chain.pop(index_chain)
                chain.insert(index_chain, "project_by_block")
                chain.insert(index_chain, "interpolate_focal_plane")
                chain.insert(index_chain, "split_in_blocks")
        except (ValueError, AttributeError):
            pass
        try:
            index_user = self.routine_names.index(RoutineName("project"))
            pipe_params = ParamsManager(self.data_manager.parameters, pipeline_type)
            if pipe_params.projection.mode == "laplacian":
                self.routine_names.pop(index_user)
                self.routine_names.insert(index_user, RoutineName("project_by_block"))
                self.routine_names.insert(
                    index_user, RoutineName("interpolate_focal_plane")
                )
                self.routine_names.insert(index_user, RoutineName("split_in_blocks"))
        except (ValueError, AttributeError):
            pass
        # Global Registration by block
        try:
            index_chain = chain.index("register_global")
            pipe_params = ParamsManager(self.data_manager.parameters, pipeline_type)
            if pipe_params.registration.alignByBlock:
                chain.pop(index_chain)
                chain.insert(index_chain, "compare_block_global")  # CompareBlockGlobal
                chain.insert(index_chain, "register_by_block")  # RegisterByBlock
        except ValueError:
            pass
        except AttributeError:
            pass
        try:
            index_user = self.routine_names.index(RoutineName("register_global"))
            pipe_params = ParamsManager(self.data_manager.parameters, pipeline_type)
            if pipe_params.registration.alignByBlock:
                self.routine_names.pop(index_user)
                self.routine_names.insert(
                    index_user, RoutineName("compare_block_global")
                )
                self.routine_names.insert(index_user, RoutineName("register_by_block"))
        except ValueError:
            pass
        except AttributeError:
            pass

        return chain

    def create_routine(
        self,
        routine_name: RoutineName,
        routine_params: Dict[
            str,
            Union[
                ProjectionParams, RegistrationParams, SegmentationParams, MatrixParams
            ],
        ],
    ):
        routine_name = routine_name.value
        routine_mapping = {
            "project": ProjectModule,
            "project_by_block": ProjectByBlockModule,
            "interpolate_focal_plane": InterpolateFocalPlane,
            "split_in_blocks": SplitInBlocks,
            "skip": routine.Skip,
            "preprocess_3d": Preprocess3D,
            "shift_2d": routine.Shift2DModule,
            "shift_3d": routine.Shift3DModule,
            "register_global": RegisterGlobalModule,
            "register_by_block": RegisterByBlock,
            "compare_block_global": CompareBlockGlobal,
            "register_local": RegisterLocal,
            "localize_2d": Localize2D,
            "fit_subpixel": FitSubpixel,
            "shift_spot_on_z": ShiftSpotOnZ,
            "segment_2d": Segment2D,
            "segment_3d": Segment3D,
            "reduce_planes": ReducePlanes,
            "extract_properties": ExtractProperties,
            "deblend_3d": Deblend3D,
            # "filter_mask": routine.FilterMaskModule,
            # "select_mask_2d": routine.SelectMask2DModule,
            # "select_mask_3d": routine.SelectMask3DModule,
            # "filter_localization": routine.FilterLocalizationModule,
            # "register_localization": routine.RegisterLocalizationModule,
            "build_trace_3d": BuildTrace3D,
            # "build_matrix": routine.BuildMatrixModule,
            "add_cycle_to_table": AddCycleToTable,
        }
        if routine_name in routine_mapping:
            print(f"> Add: {routine_name}")
            return routine_mapping[routine_name](self.data_manager, **routine_params)
        else:
            raise ValueError(f"Module {routine_name} does not exist.")

    def create_pipeline_routines(self, pipeline_type: AnalysisType, dim: int):
        print(f"\n[{pipeline_type.name} {dim}D]")
        routine_chain = self.get_routine_chain(pipeline_type, dim)
        routines: List[routine.Module] = []
        pipe_params = ParamsManager(self.data_manager.parameters, pipeline_type)
        for i in range(len(routine_chain)):
            if routine_chain[i] in self.routine_names:
                routine_params = pipe_params.get_routine_params(routine_chain[i])
                routines.append(self.create_routine(routine_chain[i], routine_params))
                # check if we don't break the chain
                if len(routines) >= 2 and not routines[-1].is_compatible(
                    routines[-2].output_type
                ):
                    raise ValueError(
                        f"Module {routine_chain[i]} cannot be used without {routine_chain[i - 1]}, for {pipeline_type} analysis."
                    )
        return routines

    def create_pipelines(self):
        print_text_inside("Creating pipelines", "=")
        print(f"Analysis types found: {self.found_analysis_types}")
        print(f"Analysis types to process: {self.user_analysis_types}")
        for analysis_type in self.found_analysis_types:
            print(f"\n[{analysis_type}]")

            if analysis_type not in self.user_analysis_types:
                continue

            for dim in self.dims:
                routines = self.create_pipeline_routines(analysis_type, dim)
                if routines:
                    self.analysis_to_process.append((analysis_type, dim))
                    print(f"> CREATED")
                    self.pipelines[analysis_type.value][dim] = Pipeline(
                        analysis_type, routines
                    )
                else:
                    print("> IGNORED")
        if not self.analysis_to_process:
            raise ValueError("No analysis to process.")

    def launch_analysis(self):
        client = None
        use_dask = self.n_threads > 1
        if use_dask:
            n_workers = find_n_workers(self.n_threads)
            if n_workers > 1:
                cluster = LocalCluster(
                    n_workers=n_workers,
                    threads_per_worker=1,
                    memory_limit="64GB",
                )
                # Create a Dask client with 4 workers
                client = Client(cluster)
                print(client.dashboard_link)
                print(
                    "$ Go to http://localhost:8787/status for information on progress..."
                )
            else:
                use_dask = False

        for analysis_type, dim in self.analysis_to_process:
            self.data_manager.refresh_input_files()
            if use_dask:
                tasks = []  # list to hold the tasks
            print_analysis_type(analysis_type, dim)
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
                if use_dask:
                    tasks.append(
                        client.submit(
                            pipe.process,
                            data_path,
                            sup_paths.pop(cycle),
                            cycle,
                        )
                    )
                else:
                    pipe.process(data_path, sup_paths.pop(cycle), cycle)
            if sup_paths:
                raise ValueError(
                    f"Supplementary data not used for analysis {analysis_type}: {sup_paths}"
                )
            # use Dask client to start the computation if use_dask is True
            if use_dask:
                client.gather(tasks)

        # Close the client
        if use_dask:
            cluster.close()
            client.close()


def find_n_workers(n_threads: int) -> int:
    """Defines the number of threads allocated"""
    n_cores = multiprocessing.cpu_count()
    max_load = 0.8
    memory_per_worker = 1200

    # we want at least 12 GB per worker
    free_m = int(os.popen("free -t -m").readlines()[1].split()[-1])

    max_n_threads = int(
        np.min(
            [
                n_cores * max_load,
                free_m / memory_per_worker,
            ]
        )
    )

    n_workers = int(np.min([max_n_threads, n_threads]))

    print(f"> Cluster with {n_workers} workers started ({n_threads} requested)")
    return n_workers
