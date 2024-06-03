# Routine description

They exist three types of routines:
1. `AbstractRoutine` is an user-friendly name for a set of routines that be link with a specific chain of routines.
2. `Routine` is a main feature - with a *load*, *run* and *save* methods - of the software callable inside a Pipeline of routines or independantly by CLI. (We also want to develop a Napari plugin for each routines.)
3. `SubRoutine` is a `Routine` class used for the internal mechanism, hidden from non-expert users. Usually, a main feature can have different way to run and one way can be enough complex to justify to split this `Routine` in different `SubRoutine`.

## AbstractRoutine list

## Routine list

| Name                 | Tested | To refactor    |
| -------------------- | ------ | -------------- |
| BuildTrace3D   | Yes    |    |
| Localize2D           | Yes    |                |
| ReducePlanes         | Yes    |                |
| ShiftSpotOnZ         | Yes    |                |
| ExtractProperties    | Yes    |                |
| FitSubpixel          | Yes    |                |
| AddCycleToTable      | Yes    |                |
| Skip           | Yes    |            |
| Shift          |        |                |
| Shift3D        | Yes    |         |
| Shift2D        | Yes    |         |
| SelectMask     |        |                |
| Project        | Yes    |         |
| RegisterGlobal | Yes    |  |
| Preprocess3D         | Yes    |                |
| RegisterLocal        | Yes    |                |
| Segment2D            | Yes    |                |
| Segment3D            | Yes    |                |
| Deblend3D            | Yes    |                |




## SubRoutine list

- SplitInBlocks
- InterpolateFocalPlane
- ProjectByBlock
- RegisterByBlock
- CompareBlockGlobal

