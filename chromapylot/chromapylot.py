from typing import List, Literal

Datatype = Literal[
    "_dapi",
    "_primer",
    "_satellite",
    "_rna",
    "_3d",
    "_2d",
    "_shift_tuple",
    "_shift_dict",
    "_shift_table",
    "_shifted",
    "_segmented",
    "_table",
    "_filtered",
    "_registered",
]
AnalysisType = Literal["fiducial", "barcode", "mask", "trace"]
ModuleName = Literal[
    "project",
    "register_global",
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
    def __init__(self, modules: List[Module]):
        self.modules = modules
        self.supplementary_data = {}

    def prepare(self):
        for module in self.modules:
            if module.supplementary_data:
                self.supplementary_data.update(module.supplementary_data)

    def assign_supplementary_data(self, module, data_path, label_name):
        for key, value in module.supplementary_data.items():
            if value is None:
                module.load_supplementary_data(data_path, label_name)
            else:
                module.supplementary_data[key] = self.supplementary_data[key]

    def choose_to_keep_data(self, module, data):
        if module.action_keyword in self.supplementary_data:
            self.supplementary_data[module.action_keyword] = data

    def choose_to_keep_input_data(self, data):
        if self.modules[0].input_type in self.supplementary_data:
            self.supplementary_data[self.modules[0].input_type] = data

    def process(self, data_path, label_name):
        data = self.modules[0].load_data(data_path, label_name)
        self.choose_to_keep_input_data(data)
        for module in self.modules:
            self.assign_supplementary_data(module, data_path, label_name)
            data = module.run(data)
            module.save_data(data_path, label_name, data)
            self.choose_to_keep_data(module, data)


class AnalysisManager:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.module_names = []
        self.pipelines = {
            "fiducial": None,
            "barcode": None,
            "mask": None,
            "trace": None,
        }

    def parse_commands(self, commands: List[str]):
        self.module_names = list(set(commands))

    def get_module_chain(self, pipeline_type: AnalysisType) -> List[ModuleName]:
        if pipeline_type == "fiducial":
            return ["project", "register_global", "shift", "register_local"]
        elif pipeline_type == "barcode":
            return ["shift", "segment", "extract"]
        elif pipeline_type == "mask":
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
            "project": ProjectModule,
            "register_global": RegisterGlobalModule,
            "shift": ShiftModule,
            "register_local": RegisterLocalModule,
            "filter_table": FilterTableModule,
            "filter_mask": FilterMaskModule,
            "segment": SegmentModule,
            "extract": ExtractModule,
            "filter_localization": FilterLocalizationModule,
            "register_localization": RegisterLocalizationModule,
            "build_trace": BuildTraceModule,
            "build_matrix": BuildMatrixModule,
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
                # check if we don't break the chain
                if len(modules) > 0 and module_chain[i - 1] not in self.module_names:
                    raise ValueError(
                        f"Module {module_chain[i]} cannot be used without {module_chain[i - 1]}, for {pipeline_type} analysis."
                    )
                modules.append(self.create_module(module_chain[i], pipeline_type))

    def create_pipelines(self):
        data_types = self.data_manager.get_data_types()
        order = ["fiducial", "barcode", "mask", "trace"]
        data_types = [x for x in order if x in data_types]
        for data_type in data_types:
            modules = self.create_pipeline_modules(data_type)
            self.pipelines[data_type] = Pipeline(modules)


# # Utilisation du pipeline
# alignment_module = AlignmentModule(method='method2') *# Utilise la méthode 2 pour l'alignement*
# modules = [PreprocessingModule(), alignment_module] *# Ajoutez vos autres modules ici*
# pipeline = ImageAnalysisPipeline(modules)

# input_paths = [...]  # Remplacez par votre liste de chemins d'entrée
# output_paths = [...]  # Remplacez par votre liste de chemins de sortie

# for input_path, output_path in zip(input_paths, output_paths):
#     for module in pipeline.modules:
#         input_data = module.load_input(input_path)
#         result = pipeline.run(input_data)
#         module.save_output(result, output_path)
