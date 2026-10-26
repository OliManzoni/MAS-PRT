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



## Statistical analysis

The [analysis folder](analysis/README.md) contains a documented Jupyter notebook and equivalent Python script for distance and trial-order mixed-effects analyses. Input columns, group labels, reference levels, outcomes and units are configurable for compatible trial-level pup-retrieval datasets. The code includes effect-size estimates and confidence intervals; datasets are supplied by the user.
