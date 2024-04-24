#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser

from chromapylot.core.core_types import CommandName, AnalysisType


def _parse_run_args(command_line_args):
    """Parse run arguments

    Returns
    -------
    ArgumentParser.args
        An accessor of run arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-C",
        "--command",
        help="Comma-separated list of module names to run.",
    )
    parser.add_argument(
        "-I",
        "--input",
        type=str,
        default=os.getcwd(),
        help="Folder path with input data.\nDEFAULT: Current directory",
    )
    parser.add_argument(
        "-O",
        "--output",
        type=str,
        default=os.getcwd(),
        help="Folder path for output data.\nDEFAULT: Current directory",
    )
    parser.add_argument(
        "-D",
        "--dimension",
        type=int,
        default=23,
        help="Dimension of input data, choice between 2, 3, 23.\nDEFAULT: 3",
    )
    parser.add_argument(
        "-A",
        "--analysis",
        type=str,
        default="all",
        help="Comma-separated list of analysis type to run (all, fiducial, barcode, trace, DAPI, primer, RNA).\nDEFAULT: all",
    )
    parser.add_argument(
        "-T",
        "--threads",
        type=int,
        default=1,
        help="Thread number to run with parallel mode.\nDEFAULT: 1 (sequential mode)",
    )
    parser.add_argument(
        "-R",
        "--ref_file",
        type=str,
        default=None,
        help="Reference file to run a module directly.\nDEFAULT: None",
    )
    parser.add_argument(
        "--in_file",
        type=str,
        default=None,
        help="Data file to run a module directly.\nDEFAULT: None",
    )
    parser.add_argument(
        "-S",
        "--sup_file",
        type=str,
        default=None,
        help="Data file to run a module directly.\nDEFAULT: None",
    )

    return parser.parse_args(command_line_args)


class RunArgs:
    """Store and check run arguments"""

    def __init__(self, command_line_args):
        parsed_args = _parse_run_args(command_line_args)
        self.commands = self.parse_commands(parsed_args.command)
        self.input = parsed_args.input
        self.output = parsed_args.output
        self.dimension = self.parse_dimension(parsed_args.dimension)
        self.analysis_types = self.parse_analysis_types(parsed_args.analysis)
        self.threads = parsed_args.threads
        self.ref_file = parsed_args.ref_file
        self.in_file = parsed_args.in_file
        self.sup_file = parsed_args.sup_file
        self._check_args()

    @classmethod
    def parse_commands(cls, command):
        """Parse command argument

        Parameters
        ----------
        command : str
            Comma-separated list of module names to run.

        Returns
        -------
        list
            List of module names to run.
        """
        return command.split(",") if command else cls._get_default_commands()

    @staticmethod
    def _get_default_commands():
        return [command.value for command in CommandName]

    @classmethod
    def parse_dimension(cls, dimension):
        """Parse dimension argument

        Parameters
        ----------
        dimension : int
            Dimension of input data, choice between 2, 3, 23.

        Returns
        -------
        List[int]
            List of dimension of input data.
        """
        if dimension == 23:
            return [2, 3]
        elif dimension == 2 or dimension == 3:
            return [dimension]
        else:
            raise ValueError(
                f"Dimension {dimension} is not available, choose between (2, 3, 23)."
            )

    @staticmethod
    def parse_analysis_types(analysis_types):
        """Parse analysis argument

        Parameters
        ----------
        analysis_types : str
            Comma-separated list of analysis type to run (all, fiducial, barcode, trace, DAPI, primer, RNA).

        Returns
        -------
        List[str]
            List of analysis type to run.
        """
        if not analysis_types or analysis_types == "all":
            return [analysis_type for analysis_type in AnalysisType]
        list_of_analysis_types = analysis_types.split(",")
        return [
            analysis_type
            for analysis_type in AnalysisType
            if analysis_type.value in list_of_analysis_types
        ]

    def _check_args(self):
        """Check run arguments"""
        if not os.path.exists(self.input):
            raise FileNotFoundError(f"Input folder {self.input} does not exist.")
        if not os.path.exists(self.output):
            raise FileNotFoundError(f"Output folder {self.output} does not exist.")
        available_commands = self._get_default_commands()
        for command in self.commands:
            if command.lower() not in available_commands:
                raise ValueError(f"Command {command} is not available.")
