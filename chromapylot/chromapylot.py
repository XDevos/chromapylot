#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
from chromapylot.core.run_args import RunArgs
from chromapylot.core.data_manager import DataManager
from chromapylot.core.analysis_manager import AnalysisManager


def main(command_line_args=None):
    begin_time = datetime.now()
    run_args = RunArgs(command_line_args)
    data_manager = DataManager(run_args)
    analysis_manager = AnalysisManager(data_manager)
    analysis_manager.parse_commands(run_args.commands)
    analysis_manager.create_pipelines(run_args.dimension)
    analysis_manager.launch_analysis()
    print("\n==================== Normal termination ====================\n")
    print(f"Elapsed time: {datetime.now() - begin_time}")


if __name__ == "__main__":
    main()
