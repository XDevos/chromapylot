#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check the non regression of RegisterLocal feature"""

import os
import shutil
import tempfile

from chromapylot.core.data_manager import extract_files

# sys.path.append("..")
from chromapylot.run_chromapylot import main
from tests.testing_tools.comparison import (
    compare_block3d_files,
    compare_line_by_line,
    compare_npy_files,
    image_pixel_differences,
)

# Build a temporary directory
tmp_dir = tempfile.TemporaryDirectory()
# Define a "register_local" directory inside the temp dir
tmp_register_local_in = os.path.join(tmp_dir.name, "register_local")
# Copy the modes & IN/OUT structure for register_local inside the "register_local" temp dir
shutil.copytree("pyhim-small-dataset/register_local/IN", tmp_register_local_in)


def template_test_register_local(
    mode: str, commands: str = "skip,preprocess_3d,shift_3d,register_local"
):
    """Check RegisterLocal feature with all possibilities"""
    inputs = os.path.join(tmp_register_local_in, mode)
    main(["-I", inputs, "-O", inputs, "-C", commands])
    generated_register_local = os.path.join(inputs, "register_local")
    reference_outputs = f"pyhim-small-dataset/register_local/OUT/{mode}/alignImages/"
    generated_files = extract_files(generated_register_local)
    reference_files = extract_files(reference_outputs)
    for filepath, short_filename, extension in generated_files:
        if "data" in filepath.split(os.sep):
            filename = f"data{os.sep}{short_filename}.{extension}"
        else:
            filename = f"{short_filename}.{extension}"
        tmp_file = os.path.join(generated_register_local, filename)
        out_file = os.path.join(reference_outputs, filename)
        assert os.path.exists(out_file)
        if extension == "npy":
            assert compare_npy_files(tmp_file, out_file)
        elif extension == "png":
            assert image_pixel_differences(tmp_file, out_file)
        elif extension == "json":
            assert compare_line_by_line(tmp_file, out_file)
        elif extension == "table" or extension == "dat":
            if mode == "alone":
                compare_block3d_files(tmp_file, out_file, atol=0.2)
            else:
                assert compare_block3d_files(tmp_file, out_file, atol=0.05)
        else:
            raise ValueError(f"Extension file UNRECOGNIZED: {filepath}")
    # check this after to have feedback if the test failed
    assert len(generated_files) == len(reference_files)


def test_with_global_done():
    template_test_register_local("with_global")


def test_without_register_global():
    template_test_register_local(
        "alone",
        commands="skip,preprocess_3d,project,register_global,shift_3d,register_local",
    )
