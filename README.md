# MAS-PRT

Machine-Automated Scoring of Pup Retrieval Test

[Instruction manual](https://mas.readthedocs.io/)


MAS is a pipeline designed to score automatically the Pup Retrieval Test. It computes automatically multiple variables, including : 

- Time to first encounter
- Time to retrieve the pup
- Distance
- Size of the Nest

More information in *DOI*


[Installation guide](https://mas.readthedocs.io/en/latest/install.html#installation)


To work, MAS uses a DeepLabCut detection of both the pup and the dam, and an automated detection of the Nest using Detectron2

## DeepLabCut 3 compatibility preview

The `codex/dlc3-compatibility` branch targets DeepLabCut 3.0.2 and Python 3.10–3.12. It includes compatibility changes for DLC 3 multi-animal inference and CSV output. The original installation instructions and releases remain the reference for the published DLC 2 workflow.

This preview has passed synthetic adapter checks. It still needs validation with the trained DLC model, nest detector, and representative videos before use for scientific analyses. See [DLC3 migration notes](DLC3_MIGRATION.md) for setup and validation details.


