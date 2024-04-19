#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Util functions for testing
"""

import copy
import numpy as np
from astropy.table import Table
from PIL import Image  # 'Image' is used to load images
from PIL import ImageChops  # 'ImageChops' for arithmetical image operations


def image_pixel_differences(base_path, compare_path):
    """
    Calculates the bounding box of the non-zero regions in the image.
    :param base_image: target image to find
    :param compare_image:  image containing the target image
    :return: The bounding box is returned as a 4-tuple defining the
             left, upper, right, and lower pixel coordinate. If the image
             is completely empty, this method returns None.
    """
    base_image = Image.open(base_path)
    compare_image = Image.open(compare_path)
    # Returns the absolute value of the pixel-by-pixel
    # difference between two images.
    diff = ImageChops.difference(base_image, compare_image)
    return not bool(diff.getbbox())


def compare_npy_files(first_file, second_file, shuffled_plans=False):
    """Load both files as numpy array and compare them.

    Args:
        first_file (string): first complete file path
        second_file (string): second complete file path

    Returns:
        bool: True if they are the same array with the same value
    """
    first_npy = np.load(first_file)
    second_npy = np.load(second_file)
    is_same = np.array_equal(first_npy, second_npy, equal_nan=True)
    if not is_same and shuffled_plans:
        reverse_first = np.swapaxes(first_npy, 0, 2)
        reverse_second = np.swapaxes(second_npy, 0, 2)
        is_same = True
        for plan in reverse_first:
            is_inside = (
                (np.equal(plan, reverse_second) | np.isnan(reverse_second))
                .all((1, 2))
                .any()
            )
            is_same = is_same and is_inside
    return is_same


def compare_ecsv_files(
    first_file, second_file, columns_to_remove: list[str] = None, shuffled_lines=False
):
    first_ecsv = Table.read(first_file, format="ascii.ecsv")
    second_ecsv = Table.read(second_file, format="ascii.ecsv")

    if columns_to_remove:
        for col in columns_to_remove:
            first_ecsv.remove_column(col)
            second_ecsv.remove_column(col)
    first_npy = first_ecsv.as_array()
    second_npy = second_ecsv.as_array()
    is_same = True
    if shuffled_lines:
        for line in first_npy:
            if not line in second_npy:
                print(f"SHUFFLE: At line {line}\n from {first_file}\n\n")
                is_same = False
                break
    else:
        comparison = first_npy == second_npy
        is_same = comparison.all()
    return is_same


def compare_block3d_files(first_file, second_file, atol=0.05):
    first_ecsv = Table.read(first_file, format="ascii.ecsv")
    second_ecsv = Table.read(second_file, format="ascii.ecsv")

    # the column with "aligned file" can be shifted in the table
    # so we need to compare the line one by one
    # We start to find the index of the column "aligned file" in the second_ecsv

    is_same = True
    for f_raw in first_ecsv:
        idx = find_index(f_raw, second_ecsv)
        if idx == -1:
            return False
        is_same = (
            is_same and f_raw["reference file"] == second_ecsv["reference file"][idx]
        )
        is_same = is_same and f_raw["blockXY"] == second_ecsv["blockXY"][idx]
        is_same = is_same and f_raw["ROI #"] == second_ecsv["ROI #"][idx]
        is_same = is_same and f_raw["label"] == second_ecsv["label"][idx]
        is_same = is_same and f_raw["block_i"] == second_ecsv["block_i"][idx]
        is_same = is_same and f_raw["block_j"] == second_ecsv["block_j"][idx]
        is_same = is_same and np.isclose(
            f_raw["shift_z"], second_ecsv["shift_z"][idx], atol=atol
        )
        is_same = is_same and np.isclose(
            f_raw["shift_x"], second_ecsv["shift_x"][idx], atol=atol
        )
        is_same = is_same and np.isclose(
            f_raw["shift_y"], second_ecsv["shift_y"][idx], atol=atol
        )
        is_same = is_same and np.isclose(
            f_raw["quality_xy"], second_ecsv["quality_xy"][idx], atol=atol
        )
        is_same = is_same and np.isclose(
            f_raw["quality_zy"], second_ecsv["quality_zy"][idx], atol=atol
        )
        is_same = is_same and np.isclose(
            f_raw["quality_zx"], second_ecsv["quality_zx"][idx], atol=atol
        )
        if not is_same:
            print(f"Block3D: At line\n {f_raw}\n diff with \n{second_ecsv[idx]}\n\n")
            return False
    return is_same


def find_index(row, table):
    for i, line in enumerate(table):
        if (
            row["aligned file"] == line["aligned file"]
            and row["block_i"] == line["block_i"]
            and row["block_j"] == line["block_j"]
        ):
            return i
    print(f"Row not found in the array:\n{row} ")
    return -1


def compare_line_by_line(first_file, second_file, shuffled_lines=False, line_start=0):
    with open(first_file, encoding="utf-8") as f_1:
        with open(second_file, encoding="utf-8") as f_2:
            f1_lines = f_1.read().splitlines()
            f2_lines = f_2.read().splitlines()
            f1_length = len(f1_lines)
            f2_length = len(f2_lines)
            line_index = 0
            is_same = f1_length == f2_length
            while is_same and (line_index < f1_length):
                if shuffled_lines:
                    is_same = f1_lines[line_index][line_start:] in [
                        f_l[line_start:] for f_l in f2_lines
                    ]
                    if not is_same:
                        print(
                            f"SHUFFLE: At line number {line_index}\n from {first_file}\n{f1_lines[line_index]}\n"
                        )
                else:
                    is_same = (
                        f1_lines[line_index][line_start:]
                        == f2_lines[line_index][line_start:]
                    )
                    if not is_same:
                        print(
                            f"At line number {line_index}\n from {first_file}\n{f1_lines[line_index]}\n from {second_file}\n{f2_lines[line_index]}"
                        )
                line_index += 1
            return is_same


def compare_trace_table(ref_file, generated_file):
    ref_table = Table.read(ref_file, format="ascii.ecsv")
    gen_table = Table.read(generated_file, format="ascii.ecsv")
    if len(ref_table) == 0 or len(ref_table) != len(gen_table):
        print(f"len: {len(ref_table)} != {len(gen_table)}")
        return False

    for i in range(len(ref_table)):
        if ref_table[i]["Spot_ID"] != gen_table[i]["Spot_ID"]:
            if ref_table[i]["Spot_ID"] in gen_table["Spot_ID"]:
                # get the index in gen_table["Spot_ID"]
                index = np.where(gen_table["Spot_ID"] == ref_table[i]["Spot_ID"])[0][0]
                # swith row inside gen_table
                deepcopy = copy.deepcopy(gen_table[i])
                gen_table[i] = gen_table[index]
                gen_table[index] = deepcopy
                print(f"Switched {i} with {index} in gen_table['Spot_ID']")
            else:
                print(f"i: {i}")
                print(
                    f"Spot_ID: {ref_table[i]['Spot_ID']} != {gen_table[i]['Spot_ID']}"
                )
                return False
        if not np.isclose(ref_table[i]["x"], gen_table[i]["x"], atol=1e-4):
            print(f"i: {i}")
            print(f"x: {ref_table[i]['x']} != {gen_table[i]['x']}")
            return False
        if not np.isclose(ref_table[i]["y"], gen_table[i]["y"], atol=1e-4):
            print(f"i: {i}")
            print(f"y: {ref_table[i]['y']} != {gen_table[i]['y']}")
            return False
        if not np.isclose(ref_table[i]["z"], gen_table[i]["z"], atol=1e-4):
            print(f"i: {i}")
            print(f"z: {ref_table[i]['z']} != {gen_table[i]['z']}")
            return False
        if ref_table[i]["Chrom"] != gen_table[i]["Chrom"]:
            print(f"i: {i}")
            print(f"Chrom: {ref_table[i]['Chrom']} != {gen_table[i]['Chrom']}")
            return False
        if ref_table[i]["Chrom_Start"] != gen_table[i]["Chrom_Start"]:
            print(f"i: {i}")
            print(
                f"Chrom_Start: {ref_table[i]['Chrom_Start']} != {gen_table[i]['Chrom_Start']}"
            )
            return False
        if ref_table[i]["Chrom_End"] != gen_table[i]["Chrom_End"]:
            print(f"i: {i}")
            print(
                f"Chrom_End: {ref_table[i]['Chrom_End']} != {gen_table[i]['Chrom_End']}"
            )
            return False
        if ref_table[i]["ROI #"] != gen_table[i]["ROI #"]:
            print(f"i: {i}")
            print(f"ROI #: {ref_table[i]['ROI #']} != {gen_table[i]['ROI #']}")
            return False
        if ref_table[i]["Mask_id"] != gen_table[i]["Mask_id"]:
            print(f"i: {i}")
            print(f"Mask_id: {ref_table[i]['Mask_id']} != {gen_table[i]['Mask_id']}")
            return False
        if ref_table[i]["Barcode #"] != gen_table[i]["Barcode #"]:
            print(f"i: {i}")
            print(
                f"Barcode #: {ref_table[i]['Barcode #']} != {gen_table[i]['Barcode #']}"
            )
            return False
        if ref_table[i]["label"] != gen_table[i]["label"]:
            print(f"i: {i}")
            print(f"label: {ref_table[i]['label']} != {gen_table[i]['label']}")
            return False
    return True
