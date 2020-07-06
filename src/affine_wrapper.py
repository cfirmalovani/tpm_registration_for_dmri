"""
DIPY wrapper for TPM affine registration.
Currently is only mask-based, will be upgraded at the future

@author: Cfir Malovani
"""

from dipy.align.imaffine import transform_centers_of_mass, MutualInformationMetric, AffineRegistration
from dipy.align.transforms import TranslationTransform3D, RigidTransform3D, AffineTransform3D
import numpy as np


class Affine_Registration:
    """
    DIPY based wrapper for TPM affine registration
    Affine registration is a prequisite for the deformable multimodal cross correlation registration
    This part is only mask-based for now, will be upgraded at the future
    """
    def __init__(self, affine_nbins = 32, 
                affine_level_iters = [10000, 1000, 100],
                affine_sigmas = [3.0, 1.0, 0.0],
                affine_factors = [4, 2, 1]):
        """
        Affine registration process initialization
        :param affine_nbins: Number of bins for the Mutual Information metric
        :param affine_level_iters: Number of iterations (per scale)
        :param affine_sigmas:  Smoothing parameters (per scale)
        :param affine_factors: Scale factors to build the scale space
        """ 
        self.affine_nbins = affine_nbins
        self.affine_level_iters = affine_level_iters
        self.affine_sigmas = affine_sigmas
        self.affine_factors = affine_factors
        
        
    def optimize_affine_registration(self, input_image, input_grid2world, 
                                     target_image, target_grid2world):
        """
        Optimizing the affine registration process
        :param input_image: Input image to be registered
        :param input_grid2world: voxel-to-space transformation of the input image
        :param target_image:  target image for registration
        :param target_grid2world: voxel-to-space transformation of the target image
        :return affine: Optimized affine transformation
        """ 
        # assert input_image.shape[-1] == 4, 'Input TPM should include 4 modalities; has {} instead'.format(input_image.shape[-1])
        # assert target_image.shape[-1] == 4, 'Target TPM should include 4 modalities; has {} instead'.format(target_image.shape[-1])
        
        metric = MutualInformationMetric(self.affine_nbins, None)
        affreg = AffineRegistration(metric = metric,
                                    level_iters = self.affine_level_iters,
                                    sigmas = self.affine_sigmas,
                                    factors = self.affine_factors,
                                    verbosity = 0)       
        c_of_mass = transform_centers_of_mass(target_image, target_grid2world,
                                              input_image, input_grid2world)
        translation = affreg.optimize(target_image, input_image, TranslationTransform3D(), None,
                                      target_grid2world, input_grid2world,
                                      starting_affine = c_of_mass.affine)        
        rigid = affreg.optimize(target_image, input_image, RigidTransform3D(), None,
                                target_grid2world, input_grid2world,
                                starting_affine = translation.affine)     
        affine = affreg.optimize(target_image, input_image, AffineTransform3D(), None,
                                 target_grid2world, input_grid2world,
                                 starting_affine = rigid.affine)
                                 
        return affine
    
    
    def apply_affine_to_multimodal_image(self, input_image, affine, is_tpm = False):
        """
        Applying an affine transformation to an input image
        :param input_image: Input image to be transformed
        :param affine: Affine transformation to be applied
        :param is_tpm: Flag for handling TPMs. If True, TPM is renormalized.
        :return output_image: Transformed image
        """ 
        self.validate_multimodal_image(input_image)
        flat_output = affine.transform(input_image[..., 0])
        output_image = np.zeros(flat_output.shape + (input_image.shape[-1],))
        output_image[..., 0] = flat_output
        for idx in range(1, input_image.shape[-1]):
            output_image[..., idx] = affine.transform(input_image[..., idx])
            
        if is_tpm:
            output_image = self.renormalize_tpm(output_image)
        
        return output_image
    
    
    def apply_inverse_affine_to_multimodal_image(self, input_image, affine, is_tpm = False):
        """
        Applying an inverse affine transformation to an input image
        :param input_image: Input image to be transformed
        :param affine: Affine transformation to be applied in reverse
        :param is_tpm: Flag for handling TPMs. If True, TPM is renormalized.
        :return output_image: Transformed image
        """ 
        self.validate_multimodal_image(input_image)
        flat_output = affine.transform_inverse(input_image[..., 0])
        output_image = np.zeros(flat_output.shape + (input_image.shape[-1],))
        output_image[..., 0] = flat_output
        for idx in range(1, input_image.shape[-1]):
            output_image[..., idx] = affine.transform_inverse(input_image[..., idx])
            
        if is_tpm:
            output_image = self.renormalize_tpm(output_image)       
            
        return output_image
    
    
    def renormalize_tpm(self, tpm):
        """
        Renormalizing a TPM for probabilistic unitary
        :param tpm: Input TPM to be renormalized
        """
        mask = np.sum(tpm[..., :-1], axis = -1)
        mask = 1 * (mask > 0.5)
        for idx in range(tpm.shape[-1] - 1):
            tpm[..., idx] *= mask
        tpm[..., -1] = 1 - mask
        
        return tpm
    
        
    def validate_multimodal_image(self, input_image):
        """
        Validating input image is indeed a 4D array
        """
        assert len(input_image.shape) == 4, 'Input image should be a 4D image'