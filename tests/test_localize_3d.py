#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check the non regression of Localize3D feature"""
import pandas as pd
import numpy as np
import os
import shutil
import tempfile

from chromapylot.core.data_manager import extract_files

# sys.path.append("..")
from chromapylot.run_chromapylot import main

# Build a temporary directory
tmp_dir = tempfile.TemporaryDirectory()
# Define a "localize_3d" directory inside the temp dir
tmp_localize_3d_in = os.path.join(tmp_dir.name, "localize_3d")
# Copy the modes & IN/OUT structure for localize_3d inside the "localize_3d" temp dir
shutil.copytree("pyhim-small-dataset/localize_3d/IN", tmp_localize_3d_in)


def template_test_localize_3d(mode: str):
    """Check Localize3D feature with all possibilities"""
    inputs = os.path.join(tmp_localize_3d_in, mode)
    main(
        [
            "-I",
            inputs,
            "-O",
            inputs,
            "-C",
            "skip,reduce_planes,preprocess_3d,shift_3d,segment_3d,deblend_3d,extract_properties,add_cycle_to_table,fit_subpixel,shift_spot_on_z",
            "-A",
            "barcode",
        ]
    )
    generated_align_images = os.path.join(inputs, "localize_3d")
    reference_outputs = f"pyhim-small-dataset/localize_3d/OUT/{mode}/segmentedObjects/"
    gen_file = (
        generated_align_images
        + os.sep
        + "data"
        + os.sep
        + "segmentedObjects_3D_barcode.dat"
    )
    ref_file = (
        reference_outputs + os.sep + "data" + os.sep + "segmentedObjects_3D_barcode.dat"
    )
    assert compare_localize_3d_files(gen_file, ref_file)


def test_full_stack_stardist():
    template_test_localize_3d("full_stack_stardist")


def test_reduce_planes_stardist():
    template_test_localize_3d("reduce_planes_stardist")


def test_reduce_planes_thresholding():
    template_test_localize_3d("reduce_planes_thresholding")


def compare_localize_3d_files(gen_file, ref_file):
    # Read the files into pandas DataFrames
    gen_df = pd.read_csv(gen_file, comment="#", delim_whitespace=True)
    ref_df = pd.read_csv(ref_file, comment="#", delim_whitespace=True)

    # Drop the 'Buid' column from both DataFrames
    gen_df = gen_df.drop(columns=["Buid"])
    ref_df = ref_df.drop(columns=["Buid"])

    # Sort by "Barcode #" and id
    gen_df = gen_df.sort_values(by=["Barcode #", "id"])
    ref_df = ref_df.sort_values(by=["Barcode #", "id"])

    # Print first line of each DataFrame with more details
    print("Generated DataFrame:")
    print(gen_df.head(1).to_string(index=False))
    print("\nReference DataFrame:")
    print(ref_df.head(1).to_string(index=False))

    # Compare the two DataFrames line by line
    diff_indices = np.where(~np.isclose(gen_df.values, ref_df.values, atol=1e-4))
    if len(diff_indices[0]) > 0:
        print("\nDifferences found:")
        for row, col in zip(diff_indices[0], diff_indices[1]):
            print(f"Row {row+1}, Column {col+1}:")
            print(f"Generated: {gen_df.iloc[row, col]}")
            print(f"Reference: {ref_df.iloc[row, col]}")
    else:
        print("\nNo differences found.")

    # Return True if all values are close, False otherwise
    return len(diff_indices[0]) == 0
