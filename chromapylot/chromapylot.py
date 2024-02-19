class ImageAnalysisPipeline:
    def __init__(self, modules):
        self.modules = modules

    def run(self, input_data):
        for module in self.modules:
            input_data = module.process(input_data)
        return input_data


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
