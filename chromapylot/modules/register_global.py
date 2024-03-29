#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
from chromapylot.modules.module import Module
from chromapylot.core.core_types import DataType
from chromapylot.parameters.registration_params import RegistrationParams
import numpy as np
import os


class RegisterGlobalModule(Module):
    def __init__(self, registration_params: RegistrationParams):
        super().__init__(
            input_type=DataType.IMAGE_2D,
            output_type=DataType.SHIFT_TUPLE,
            reference_type=DataType.IMAGE_2D,
        )
        self.reference_data = None
        self.ref_fiducial = registration_params.referenceFiducial

    def run(self, image):
        return [0, 0]

    def load_data(self, input_path, in_dir_length):
        print(f"[Load] {self.input_type.value}")
        short_path = input_path[in_dir_length:]
        print(f"> $INPUT{short_path}")
        return np.load(input_path)

    def save_data(self, data, output_dir, input_path):
        print("Saving shift tuple.")

    def load_reference_data(self, paths: List[str]):
        good_path = None
        for path in paths:
            if self.ref_fiducial in os.path.basename(path):
                good_path = path
                break
        self.reference_data = np.load(good_path)
