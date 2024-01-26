# Tissue Probability Based Registration for Diffusion Weighted MRI

# Introduction
Multivariate approach for MRI registration using calculated tissue probability maps (TPM) of the brain, based on the work done in [Ido Tavor's Lab](https://www.tau.ac.il/~idotavor/) and [The Lab for Advanced MRI](https://beneliezer-lab.com/).

This repository incorporates the source code and (in the future) usable pipeline of this method.

keywords: Diffusion MRI, Tissue Segmentation, Multimodal, Registration

## Recent Updates

 - Uploaded textual UI for the DTI registration pipeline.

# Setup
## Creating Python Enviroment
- Using PIP: 

`pip install -r requirements.txt`

- Using Conda: 

```
conda env create -f tpm_reg_env.yml -n tpm_reg
conda activate tpm_reg
```

## Build Multimodal Cross-Correlation Module
- Run the following script
```
cd multimodal_crosscorr
python setup.py build_ext --inplace
```
- Copy created .pyd/.so file to src/mod_dipy




# TPM Prediction and Registration
Currently only CLI is available for the DTI pipeline. 
UI is implemented using the `argparse` module.

Use [`src/tpm_reg_pipeline.py`](src/tpm_reg_pipeline.py) to run current CLI.
```
usage: TPM Registration [-h] -i FA_filename [-r ref_TPM_filename]
                        [-o warped_filename] [-ot [TPM_savedir]]
                        [-m mask_filename] [-sw [warp_savedir]]
                        [--smoothing [parameter [parameter ...]]] [-v]

  -i FA_filename, --input FA_filename
                        filename of input FA map - must be in the same
                        directory with Lambda maps
  -r ref_TPM_filename, --ref ref_TPM_filename, --tpm ref_TPM_filename
                        reference for TPMs of the target space (default is
                        MNI)
  -o warped_filename, --out warped_filename
                        destination folder for output (warped) DTI maps
                        (default is source directory)
  -ot [TPM_savedir], --out_tpm [TPM_savedir]
                        if activated, saving calculated TPM in destination
                        folder (default is source directory)
  -m mask_filename, --mask mask_filename
                        binary mask file (default will assume the DTI maps are
                        already masked)
  --interpolation interpolation_method, --interp interpolation_method
                        interpolation method for transformation: linear (default) / nearest
  -sw [warp_savedir], --save_wrap [warp_savedir]
                        if activated, saving warp fields in destination folder
                        (default is source directory)
  --smoothing [parameter [parameter ...]]
                        if activated, gaussian smoothing parameters for TPM -
                        may insert values for window size and sigma (default
                        are 5, 1)
  -v, --verbose         if activated, provide status reports of the pipeline
```

# Applying Precalculated warp fields
Use [`src/tpm_reg_applywarp.py`](src/tpm_reg_applywarp.py) to run current CLI.
```
usage: TPM Registration Applywarp [-h] -i MRI scan to register -w Saved warp fields [-o Output path for warped scan] [--inverse] [--interpolation interpolation_method]

  -i MRI scan to register, --input MRI scan to register
                        A scan file to be registered
  -w Saved warp fields, --saved_warps Saved warp fields
                        Saved warped fields to be applied
  -o Output path for warped scan, --out Output path for warped scan
                        destination folder for output (warped) DTI maps (default is source directory)
  --inverse             Apply the inverse transformation on input
  --interpolation interpolation_method, --interp interpolation_method
                        interpolation method for transformation: linear (default) / nearest

```
# Saved Warp Fields
The calculated warp fields are saved in two formats:
* Binary (pickled) .SAV file - Used during runtime. This file contains 3 objects:
    * The source grid-to-world coordinates affine matrix.
    * The target grid-to-world coordinates affine matrix.
    * A tuple containing binary objects of the affine + deformable transformations.
* MAT file - For debugging purposes and MATLAB integration. This file contains:
    * `source_grid2world` - The source grid-to-world coordinates affine matrix.
    * `target_grid2world` - The target grid-to-world coordinates affine matrix.
    * `affine_transform_forward` - The affine transformation matrix.
    * `affine_transform_backward` - The affine inverse transformation matrix.
    * `deform_mapping_forward` - Shift maps for the deformable transformation.
    * `deform_mapping_backward` - Shift maps for the inverse deformable transformation.

## Reference
Malovani, C., Friedman, N., Ben-Eliezer, N. and Tavor, I. (2021), Tissue Probability Based Registration of Diffusion-Weighted Magnetic Resonance Imaging. J Magn Reson Imaging, 54: 1066-1076. https://doi.org/10.1002/jmri.27654
