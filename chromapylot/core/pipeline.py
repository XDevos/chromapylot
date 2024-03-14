from typing import List, Dict, Any
from chromapylot.core.data_manager import DataManager
import modules.module as mod
from chromapylot.core.core_types import DataType, AnalysisType


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
