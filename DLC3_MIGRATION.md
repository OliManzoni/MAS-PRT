# MAS adaptation for DeepLabCut 3.0.2 — validation candidate

This local branch targets the published DeepLabCut 3.0.2 wheel (its metadata and analyze_videos signature were inspected). The development branch advertises 3.0.3, which was not available from the package index during verification.

## Installation in a separate environment
Use Python 3.10–3.12 (3.12 used for adapter checks). Configure PyTorch for the machine following the official PyTorch instructions, then install this source with `python -m pip install .`. Install Detectron2 separately following its official build instructions for that PyTorch/platform combination. Tkinter is required for the MAS GUI. The complete dependency environment has not yet been resolved/installed and tested together.

DLC's TensorFlow extra is included to support legacy trained projects. Keep the original project directory, configuration and snapshots together. Set MAS_DLC_CONFIG to the trained project's config.yaml and MAS_NEST_MODEL to the existing Detectron2 nest-model directory before launching `python -m MAS`. Copy or retain your existing downloaded models; package imports no longer download or initialize them automatically. TensorFlow weights are not converted into PyTorch weights by installing DLC3. Engine selection is delegated to DLC's trained-shuffle metadata.

DLC3.0.2 declares pandas>=2.2,<3. This candidate consequently uses pandas 2 and conservative NumPy<2 for legacy TensorFlow/imgaug models. It does not claim pandas3 or NumPy2 compatibility for the entire inference stack.

## Changes
- Fixed package metadata, executable entry points and packaged layout YAML.
- Replaced forced TensorFlow2.10 and older conflicting binary pins with DLC3's TensorFlow extra and compatible numerical ranges; removed redundant OpenCV installations.
- Removed import-time downloads/GPU initialization; explicit external model paths are supported.
- Video API receives a list, a created output directory and automatic multi-animal tracking. Exceptions propagate instead of being silently skipped.
- Label generation reads predictions from the actual destination without copying and deleting user files.
- CSVs with three/four header rows are mapped by anatomical names into the existing scoring order. Missing/ambiguous parts and nonconsecutive frame indices are rejected.
- Platform-neutral paths, lowercase GUI import, preserved underscores in video names; scoring errors now stop processing with context.

## Validation and limits
Adapter tests cover shuffled CSV columns, both header layouts, missing frame rejection, video-call arguments and error propagation with a simulated DLC API. Synthetic scoring tests exercise means, missing-frame interpolation, first encounter and distance. They pass in Python3.12/pandas2.3.3/NumPy1.26.4 and in the separate pandas3.1.1/NumPy2.5.5 numerical-only environment. Tests do not execute real DLC, TensorFlow or Detectron2 inference. A wheel build validates packaging, not installation of the full GPU stack.

Before replacing the working analysis installation, run a representative original video with the same trained models, nest settings and reference tracking outputs. Compare frame indices, tracking coordinates and all derived distances/times. No real-video scientific regression or GUI/GPU validation has been performed. Existing statistical analysis and J-column filtering are separate and were not changed.

Sources: https://deeplabcut.github.io/DeepLabCut/docs/installation.html and the downloaded deeplabcut-3.0.2 wheel metadata/API source. This document supersedes the earlier compatibility-review snapshot for this branch.
