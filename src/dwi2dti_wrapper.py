"""
DIPY wrapper for creating DTI images from DWI scans

@author: Cfir Malovani
"""

import numpy as np
import nibabel as nib

from dipy.reconst.dti import TensorModel, fractional_anisotropy
from dipy.segment.mask import median_otsu
from dipy.data import gradient_table


class DTI_Image_Creator:
    """
    DIPY based tool for the creation of DTI images to feed into the tissue classifier
    """
    def __init__(self, dwi_file, bvecs_file, bvals_file, mask_file = None):
        """
        Initialization of the creator:
        :param dwi_file: Path of DWI scan file
        :param bvecs_file: Path of b-vectors file
        :param bvals_file: Path of b-values file
        :param mask_file: (Optional) path of mask file for the DWI scan
        """
        self.dwi_file = dwi_file
        self.bvecs_file = bvecs_file
        self.bvals_file = bvals_file
        self.mask_file = mask_file   
        
        self.gradients = None
        
    
    def build_gradients_table(self):
        """
        Loads b-vectors and b-values into memory, 
        and creates the gradient table needed for DTI calculation
        """
        # Loading scan files
        f = open(self.bvals_file)
        bvals_list = f.read().split()
        f.close()
        
        f = open(self.bvecs_file)
        bvecs_list = f.read().split()
        f.close()       

        # Transform string lists to numpy arrays
        bvals = np.asarray([float(i) for i in bvals_list])
        bvecs = np.asarray([[float(i), float(j), float(k)] for i,j,k \
                 in zip(bvecs_list[0::3], bvecs_list[1::3], bvecs_list[2::3])])
            
        # Normalizing bvecs
        for idx in range(bvecs.shape[0]):
            normal_factor = np.linalg.norm(bvecs[idx, :])
            if normal_factor > 0:
                bvecs[idx, :] = bvecs[idx, :] / normal_factor
                        
        self.gradients = gradient_table(bvals, bvecs)
    
    
    def load_dwi_scan(self):
        """
        Loads the DWI scan into memory
        :return masked_dwi_image: DWI image after skull-stripping
        :return mask_image: brain mask (either from file or calculated)
        :return dwi_affine: Voxel-to-space transformation of the DWI image
        """
        dwi_scan = nib.load(self.dwi_file)
        dwi_image = dwi_scan.get_fdata()
        dwi_affine = dwi_scan.affine
        if self.mask_file is not None: # Using available mask file
            mask_image = nib.load(self.mask_file).get_fdata()
            masked_dwi_image = dwi_image * np.repeat(mask_image[..., np.newaxis], dwi_image.shape[-1], axis = -1)
        else: # Calculating a mask using DIPY's median_otsu
            masked_dwi_image, mask_image = median_otsu(dwi_image, vol_idx=range(dwi_image.shape[-1]), median_radius=4,
                                              numpass=4, autocrop=False, dilate=3)
        
        return masked_dwi_image, mask_image, dwi_affine
    
    
    def calculate_dti_image(self):
        """
        Calculate of the DTI image
        :return dti_image: The calculated DTI image - a 4D image containing
                           FA, L1, L2, L3
        :return mask_image: brain mask (either from file or calculated)
        :return dti_affine: Voxel-to-space transformation of the DTI image
        """
        # Load DWI scan
        masked_dwi_image, mask_image, dti_affine = self.load_dwi_scan()
        
        # Create gradients table
        self.build_gradients_table()
        
        # Calculating DTI parameters
        tensor_model = TensorModel(self.gradients)
        tensor_fit = tensor_model.fit(masked_dwi_image)  
        FA = fractional_anisotropy(tensor_fit.evals)
        FA[np.isnan(FA)] = 0
        FA[FA > 1] = 1
        
        # Building DTI array
        dti_image = np.zeros(FA.shape + (4,))
        dti_image[..., 0] = FA
        dti_image[..., 1:] = tensor_fit.evals
        dti_image[dti_image < 0] = 0
        
        return dti_image, mask_image, dti_affine