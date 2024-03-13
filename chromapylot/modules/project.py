#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from core_types import DataType
from module import Module
from parameters import ProjectionParams


class ProjectModule(Module):
    def __init__(self, projection_params: ProjectionParams):
        super().__init__(input_type=DataType.IMAGE_3D, output_type=DataType.IMAGE_2D)
        self.mode = projection_params.mode

    def run(self, array_3d):
        if self.mode == "laplacian":
            pass
