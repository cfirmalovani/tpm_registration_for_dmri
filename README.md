# Tissue Probability Based Registration for Diffusion Weighted MRI

# Introduction
Multivariate approach for MRI registration using calculated tissue probability maps (TPM) of the brain, based on the work done in [Ido Tavor's Lab](https://www.tau.ac.il/~idotavor/) and [The Lab for Advanced MRI](https://beneliezer-lab.com/).

This repository incorporates the source code and (in the future) usable pipeline of this method.

keywords: Diffusion MRI, Tissue Segmentation, Multimodal, Registration

## Recent Updates

 - Uploaded textual UI for the DTI registration pipeline.

# Instructions

## Setup
This implementation is based on several existing python packages:
- [`DIPY`](https://dipy.org/)
- [`NiBabel`](https://nipy.org/nibabel/)
- [`NumPy`](https://numpy.org/)
- [`XGBoost`](https://xgboost.readthedocs.io/)

Installing these packages are a prerequisite for using this pipeline.

Otherwise, the current implementation supports Windows, MacOS and Linux. 

## TPM Prediction and Registration
Currently only textual UI is available for the DTI pipeline. 
UI is implemented using the `argparse` module.

Run `python tpm_reg_pipeline.py -h` for further details.
