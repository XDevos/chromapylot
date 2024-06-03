# Runtime algorithm description

There is four nested levels.

## Main level

1. Parsing user arguments
2. Make the inventory of the input data
3. Convert the user commands to explicit routines
4. Preparation of the parameters
5. Creation of routine chain for each analysis type
6. Initialization of Dask cluster
7. Analysis launching:

## Analysis level

1. With a refreshing of input data
2. Reference data loading
3. Research the supplementary data un-generated during the pipeline
4. Find the path of un-generated supplementary data and affect them for each cycle
5. Find the path for each cycle of the input data of the first routine
6. Process the pipeline for each cycle:

## Pipeline level

1. Load the input data
2. [Optional] Affect this input data to a supplementary data for a next routine
3. Affect un-generated supplementary data paths to the Pipeline attribute that manage the all supplementary data of its routines.
4. Process sequentially each routines:

## Routine level

1. Load the supplementary data
2. Run
3. Save output data and its optional other visualization or comparison  with the input(s)
4. [Optional] Affect this output data to a supplementary data for a next routine
5. Affect output data to the next input data







