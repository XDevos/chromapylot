import json
import os

import numpy as np
from tifffile import imread, imsave
from astropy.table import Table

from .shift_module import shift_3d_array_subpixel
from .extract_module import extract_properties


def get_img_name(label_name):
    return f"scan_002_{label_name}_001_ROI_converted_decon_ch01"


def get_file_path(directory, filename, extension):
    return os.path.join(directory, f"{filename}.{extension}")


def load_shifts_from_json(file_path):
    with open(file_path, "r") as file:
        shifts = json.load(file)
    return shifts["ROI:001"]


def skip_z_planes(array_3d):
    if not isinstance(array_3d, np.ndarray) or len(array_3d.shape) != 3:
        raise ValueError("Input must be a 3D numpy array.")

    return array_3d[::2, :, :]


def shift_and_skip(input_path, output_path, label_name="mask28"):
    img_name = get_img_name(label_name)
    image_path = get_file_path(input_path, img_name, "tif")
    image = imread(image_path)
    shifts = load_shifts_from_json(get_file_path(input_path, "shifts", "json"))
    skip_img = skip_z_planes(image)
    shifted_image = shift_3d_array_subpixel(skip_img, shifts[label_name])
    imsave(get_file_path(output_path, img_name + "_shifted", "tif"), shifted_image)


def extract_to_ecsv_table(input_path, output_path, label_name="mask28"):
    img_name = get_img_name(label_name)
    shifted_skipped_img_path = get_file_path(output_path, img_name + "_shifted", "tif")
    mask_path = get_file_path(input_path, img_name + "_3Dmasks", "npy")
    image = imread(shifted_skipped_img_path)
    masks = np.load(mask_path)

    properties = extract_properties(image, masks)
    properties_table = Table(properties)
    properties_table.sort("max_intensity", reverse=True)
    properties_table.write(
        get_file_path(output_path, img_name + "_props", "ecsv"),
        format="ascii.ecsv",
        overwrite=True,
    )


def filter_ecsv_by_max_intensity(
    input_path, output_path, label_name="mask28", threshold=10000
):
    img_name = get_img_name(label_name)
    props_path = get_file_path(output_path, img_name + "_props", "ecsv")
    properties_table = Table.read(props_path, format="ascii.ecsv")
    filtered_table = properties_table[properties_table["max_intensity"] > threshold]
    filtered_table.write(
        get_file_path(output_path, img_name + f"_props_filtered_{threshold}", "ecsv"),
        format="ascii.ecsv",
        overwrite=True,
    )


def filter_mask_by_label(input_path, output_path, label_name="mask28", threshold=10000):
    img_name = get_img_name(label_name)
    mask_path = get_file_path(input_path, img_name + "_3Dmasks", "npy")
    masks = np.load(mask_path)
    props_path = get_file_path(
        output_path, img_name + f"_props_filtered_{threshold}", "ecsv"
    )
    filtered_table = Table.read(props_path, format="ascii.ecsv")
    filtered_mask = np.zeros_like(masks)
    for label in filtered_table["label"]:
        filtered_mask[masks == label] = label
    np.save(
        get_file_path(output_path, img_name + f"_mask_filtered_{threshold}", "npy"),
        filtered_mask,
    )


def main():
    base = "/mnt/grey/DATA/users/xdevos/oversegment_Christel/normal_filtered_mask_1500/"
    input_path = os.path.join(base, "INPUT")
    output_path = os.path.join(base, "mask_3d", "data")

    for label_name in ["mask28", "mask639", "mask708"]:
        shift_and_skip(input_path, output_path, label_name)
        extract_to_ecsv_table(input_path, output_path, label_name)
        thr = 1500
        filter_ecsv_by_max_intensity(input_path, output_path, label_name, threshold=thr)
        filter_mask_by_label(input_path, output_path, label_name, threshold=thr)


if __name__ == "__main__":
    main()
