import numpy as np
from astropy.table import Table

from .main import get_img_name, get_file_path


class FilterTableModule(Module):
    def __init__(self, props={"max_intensity": 1500}) -> None:
        super().__init__()
        self.props = props

    def run(self, table):
        """Filter the table based on the properties given in the props dictionary."""
        for key, value in self.props.items():
            table = table[table[key] > value]
        return table

    def load_data(self, input_path, label_name):
        img_name = get_img_name(label_name)
        props_path = get_file_path(input_path, img_name, "ecsv")
        properties_table = Table.read(props_path, format="ascii.ecsv")
        return properties_table

    def save_data(self, output_path, label_name, data):
        img_name = get_img_name(label_name)
        data.write(
            get_file_path(output_path, img_name + "_filtered", "ecsv"),
            format="ascii.ecsv",
            overwrite=True,
        )


class FilterMaskModule(Module):
    def __init__(self) -> None:
        super().__init__(supplementary_data={"_3Dmasks": None})

    def run(self, mask, table):
        filtered_mask = np.zeros_like(mask)
        for label in table["label"]:
            filtered_mask[mask == label] = label
        return filtered_mask

    def load_supplementary_data(self, input_path, label_name):
        img_name = get_img_name(label_name)
        mask_path = get_file_path(input_path, img_name + "_3Dmasks", "npy")
        self.supplementary_data["_3Dmasks"] = np.load(mask_path)

    def load_data(self, input_path, label_name):
        img_name = get_img_name(label_name)
        props_path = get_file_path(input_path, img_name + "_filtered", "ecsv")
        properties_table = Table.read(props_path, format="ascii.ecsv")
        return properties_table

    def save_data(self, output_path, label_name, data):
        img_name = get_img_name(label_name)
        np.save(
            get_file_path(output_path, img_name + "_mask_filtered", "npy"),
            data,
        )
