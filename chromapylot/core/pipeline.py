from typing import Any, Dict, List

from chromapylot.routines import routine

from chromapylot.core.core_types import AnalysisType, DataType
from chromapylot.core.data_manager import DataManager
from chromapylot.core.core_logging import print_text_inside


class Pipeline:
    def __init__(
        self,
        analysis_type: AnalysisType,
        routines: List[routine.Module],
    ):
        self.analysis_type = analysis_type
        self.routines = routines
        self.supplementary_data: Dict[DataType, Any] = {}

    def prepare(self, data_manager: DataManager):
        first_input_type = self.routines[0].input_type
        generated_data_type: List[DataType] = [first_input_type]
        supplementary_data_to_find = []
        for routine in self.routines:
            if routine.reference_type is not None:
                # Load data to keep during all processes
                print(
                    f"Loading reference data for {routine.__class__.__name__} with {self.analysis_type} and {routine.reference_type}."
                )
                paths = data_manager.get_paths_from_analysis_and_data_type(
                    self.analysis_type, routine.reference_type
                )
                routine.load_reference_data(paths)
            # Collect data type to keep during one process
            sup_type = routine.supplementary_type
            if sup_type is not None:
                if isinstance(sup_type, list):
                    supp_data_found = False
                    for i in range(len(sup_type)):
                        if not supp_data_found and sup_type[i] in generated_data_type:
                            supp_data_found = True
                            self.supplementary_data[sup_type[i]] = None
                            print(f"Replace {sup_type} by {sup_type[i]}.")
                            sup_type = sup_type[i]
                    if not supp_data_found:
                        supplementary_data_to_find.append(sup_type)
                else:
                    self.supplementary_data[sup_type] = None
                    if sup_type not in generated_data_type:
                        supplementary_data_to_find.append(sup_type)
            generated_data_type.append(routine.output_type)
        return first_input_type, supplementary_data_to_find

    def choose_to_keep_data(self, routine, data):
        if routine.output_type in self.supplementary_data:
            self.supplementary_data[routine.output_type] = data

    def choose_to_keep_input_data(self, data):
        if self.routines[0].input_type in self.supplementary_data:
            self.supplementary_data[self.routines[0].input_type] = data

    def update_supplementary_data(self, supplementary_paths):
        for key, value in supplementary_paths.items():
            if key not in self.supplementary_data:
                raise ValueError(f"Supplementary data type {key} not found.")
            elif self.supplementary_data[key] is not None:
                raise ValueError(f"Supplementary data type {key} already exists.")
            else:
                self.supplementary_data[key] = value

    def load_supplementary_data(self, routine: routine.Module, cycle: str):
        data_type = routine.supplementary_type
        if data_type:
            if data_type == DataType.CYCLE:
                return cycle
            elif data_type in self.supplementary_data:
                if self.supplementary_data[data_type] is None:
                    return routine.load_supplementary_data(None, cycle)
                elif isinstance(self.supplementary_data[data_type], str):
                    return routine.load_supplementary_data(
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
        supplementary_paths: Dict[DataType, str],
        cycle: str,
    ):
        print_text_inside(cycle, ".")
        input_data = self.routines[0].load_data(data_path)
        self.choose_to_keep_input_data(input_data)
        self.update_supplementary_data(supplementary_paths)
        for routine in self.routines:
            routine.print_routine_name()
            supplementary_data = self.load_supplementary_data(routine, cycle)
            if routine.switched:
                input_data, supplementary_data = supplementary_data, input_data
            if supplementary_data is None:
                output = routine.run(input_data)
            else:
                output = routine.run(input_data, supplementary_data)
            routine.save_data(output, data_path, input_data, supplementary_data)
            self.choose_to_keep_data(routine, output)
            input_data = output
