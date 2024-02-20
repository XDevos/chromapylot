class ImageAnalysisPipeline:
    def __init__(self, modules):
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

    def run(self, data_path, label_name):
        data = self.modules[0].load_data(data_path, label_name)
        for module in self.modules:
            self.assign_supplementary_data(module, data_path, label_name)
            data = module.run(data)
            module.save_data(data_path, label_name, data)
            self.choose_to_keep_data(module, data)
        return data


# Utilisation du pipeline
alignment_module = AlignmentModule(method='method2') *# Utilise la méthode 2 pour l'alignement*
modules = [PreprocessingModule(), alignment_module] *# Ajoutez vos autres modules ici*
pipeline = ImageAnalysisPipeline(modules)

input_paths = [...]  # Remplacez par votre liste de chemins d'entrée
output_paths = [...]  # Remplacez par votre liste de chemins de sortie

for input_path, output_path in zip(input_paths, output_paths):
    for module in pipeline.modules:
        input_data = module.load_input(input_path)
        result = pipeline.run(input_data)
        module.save_output(result, output_path)
