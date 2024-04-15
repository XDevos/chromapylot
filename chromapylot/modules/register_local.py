#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from chromapylot.modules.module import Module
from chromapylot.parameters.registration_params import RegistrationParams
from chromapylot.core.core_types import DataType


class RegisterLocal(Module):
    def __init__(self, registration_params: RegistrationParams):
        super().__init__(
            input_type=DataType.IMAGE_3D_SHIFTED,
            output_type=DataType.REGISTRATION_TABLE,
            reference_type=DataType.IMAGE_3D,
            supplementary_type=None,
        )

    def run(self, data, supplementary_data=None):
        raise NotImplementedError

    def load_data(self, input_path):
        raise NotImplementedError

    def save_data(self, data, input_path):
        raise NotImplementedError

    def load_reference_data(self, paths: List[str]):
        raise NotImplementedError

    def load_supplementary_data(self, input_path, cycle):
        """
        Load supplementary data for the module.

        Args:
            input_path (str): The path or the directory to the input data.
            cycle (str): The cycle to process.

        Returns:
            Any: The supplementary data.
        """
        raise NotImplementedError
