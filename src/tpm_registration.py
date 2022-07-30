"""
An implementation of the TPM based multimodal registration

@author: Cfir Malovani
"""

from mod_dipy.imwarp import SymmetricDiffeomorphicRegistration
from mod_dipy.metrics import MCC_Metric
from affine_wrapper import Affine_Registration
import numpy as np
import pickle


class TPM_Registration:
    """
    TPM based registration of dMRI images, using multimodal cross correlation (MCC)
    """

    def __init__(self, affine_nbins=32,
                 affine_sampling_prop=None,
                 affine_level_iters=[10000, 1000, 100],
                 affine_sigmas=[3.0, 1.0, 0.0],
                 affine_factors=[4, 2, 1],
                 deformable_level_iters=[20, 20, 20], verbose=False):
        """
        TPM based registration process initialization
        :param affine_nbins: Number of bins for the Mutual Information metric used by affine registraion
        :param affine_level_iters: Number of iterations (per scale) for affine registraion
        :param affine_sigmas:  Smoothing parameters (per scale) for affine registraion
        :param affine_factors: Scale factors to build the scale space for affine registraion
        :param deformable_level_iters: Number of iterations (per scale) for deformable registraion
        :param verbose: Verbosity flag
        """
        self.deformable_level_iters = deformable_level_iters
        self.affine_reg = Affine_Registration(affine_nbins, affine_sampling_prop,
                                              affine_sigmas, affine_factors)
        self.verbose = verbose

        # Other members are initialized later during optimization process
        self.start_grid2world = None
        self.target_grid2world = None
        self.registration_methods = None

    def optimize_deformable_registration(self, input_image, target_image):
        """
        MCC based deformable registration 
        :param input_image: 4D input image to be registered. 
                            Input image should be pre-aligned to the target image via affine registration
        :param target_image: 4D target image for registration
        :return deform_mapping:  Optimized deformation fields
        """
        assert input_image.shape == target_image.shape, 'Input shape should match target shape'
        assert len(input_image.shape) == 4, 'Images should be 4D arrays'

        sdr = SymmetricDiffeomorphicRegistration(MCC_Metric(3), self.deformable_level_iters)
        deform_mapping = sdr.optimize(target_image, input_image)

        return deform_mapping

    def optimize_tpm_registration(self, input_image, input_grid2world,
                                  target_image, target_grid2world):
        """
        Optimizing the full TPM registration process (affine + deformable)
        :param input_image: Input image to be registered
        :param input_grid2world: voxel-to-space transformation of the input image
        :param target_image:  target image for registration
        :param target_grid2world: voxel-to-space transformation of the target image
        """
        self.start_grid2world = input_grid2world
        self.target_grid2world = target_grid2world

        input_mask = 1 - input_image[..., -1]
        target_mask = 1 - target_image[..., -1]
        self.print_status('Optimizing affine transformation')
        affine = self.affine_reg.optimize_affine_registration(input_mask, input_grid2world,
                                                              target_mask, target_grid2world)
        warped_input_image = self.affine_reg.apply_affine_to_multimodal_image(input_image, affine, is_tpm=True)
        self.print_status('Optimizing multimodal deformable transformation')
        deform_mapping = self.optimize_deformable_registration(warped_input_image, target_image)
        self.registration_methods = (affine, deform_mapping)

    def register_scan(self, input_scan, input_grid2world, is_tpm=False):
        """
        Applying optimized transformations on a given image
        :param input_image: Input image to be transformed (3D or 4D)
        :param input_grid2world: Voxel-to-space transformation of the input image
        :param is_tpm: Flag for handling TPMs. If True, TPM is renormalized.
        :return Transformed image by optimized transformation fields
        """
        num_dimensions = len(input_scan.shape)
        assert num_dimensions == 4 or num_dimensions == 3, 'Input scan should be either 3D or 4D'
        if num_dimensions == 4:
            return self.register_4d_scan(input_scan, input_grid2world, is_tpm)
        elif num_dimensions == 3:
            return self.register_3d_scan(input_scan, input_grid2world)

    def inverse_register_scan(self, input_scan, input_grid2world, is_tpm=False):
        """
        Applying optimized inverse transformations on a given image
        :param input_image: Input image to be transformed (3D or 4D)
        :param input_grid2world: Voxel-to-space transformation of the input image
        :param is_tpm: Flag for handling TPMs. If True, TPM is renormalized.
        :return Transformed image by optimized transformation fields
        """
        num_dimensions = len(input_scan.shape)
        assert num_dimensions == 4 or num_dimensions == 3, 'Input scan should be either 3D or 4D'
        if num_dimensions == 4:
            return self.inverse_register_4d_scan(input_scan, input_grid2world, is_tpm)
        elif num_dimensions == 3:
            return self.inverse_register_3d_scan(input_scan, input_grid2world)

    def register_4d_scan(self, input_scan, input_grid2world, is_tpm=False):
        """
        Applying optimized transformations on a given 4D image
        :param input_image: Input 4D image to be transformed
        :param input_grid2world: Voxel-to-space transformation of the input image
        :param is_tpm: Flag for handling TPMs. If True, TPM is renormalized.
        :return warped_scan: Transformed image by optimized transformation fields
        """
        self.validate_registration_conditions(input_grid2world)
        affine, mapping = self.registration_methods
        transformed_scan = self.affine_reg.apply_affine_to_multimodal_image(input_scan, affine, is_tpm)
        warped_scan = np.zeros(transformed_scan.shape)
        for idx in range(input_scan.shape[-1]):
            warped_scan[..., idx] = np.squeeze(mapping.transform(np.expand_dims(transformed_scan[..., idx], axis=-1)))

        if is_tpm:
            warped_scan = self.renormalize_tpm(warped_scan)

        return warped_scan

    def inverse_register_4d_scan(self, input_scan, input_grid2world, is_tpm=False):
        """
        Applying optimized inverse transformations on a given 4D image
        :param input_image: Input 4D image to be inverse
        :param input_grid2world: Voxel-to-space transformation of the input image
        :param is_tpm: Flag for handling TPMs. If True, TPM is renormalized.
        :return warped_scan: Transformed image by optimized transformation fields
        """
        self.validate_registration_conditions(input_grid2world, is_inverse=True)
        affine, mapping = self.registration_methods
        warped_scan = np.zeros(input_scan.shape)
        for idx in range(input_scan.shape[-1]):
            warped_scan[..., idx] = np.squeeze(mapping.transform_inverse(np.expand_dims(input_scan[..., idx], axis=-1)))
        transformed_scan = self.affine_reg.apply_inverse_affine_to_multimodal_image(warped_scan, affine, is_tpm)

        if is_tpm:
            warped_scan = self.renormalize_tpm(warped_scan)

        return transformed_scan

    def register_3d_scan(self, input_scan, input_grid2world):
        """
        Applying optimized transformations on a given 3D image
        :param input_image: Input 4D image to be transformed
        :param input_grid2world: Voxel-to-space transformation of the input image
        :return warped_scan: Transformed image by optimized transformation fields
        """
        return self.register_4d_scan(input_scan[..., np.newaxis], input_grid2world)[..., 0]

    def inverse_register_3d_scan(self, input_scan, input_grid2world):
        """
        Applying optimized inverse transformations on a given 3D image
        :param input_image: Input 4D image to be inverse
        :param input_grid2world: Voxel-to-space transformation of the input image
        :return warped_scan: Transformed image by optimized transformation fields
        """
        return self.inverse_register_4d_scan(input_scan[..., np.newaxis], input_grid2world)[..., 0]

    def renormalize_tpm(self, tpm):
        """
        Renormalizing a TPM for probabilistic unitary
        :param tpm: Input TPM to be renormalized
        """
        mask = np.sum(tpm[..., :-1], axis=-1)
        mask = 1 * (mask > 0.5)
        for idx in range(tpm.shape[-1] - 1):
            tpm[..., idx] *= mask
        tpm[..., -1] = 1 - mask

        return tpm

    def save_warp_fields(self, filepath):
        """
        Save optimized warp fields and affine matrices to file.
        :param filepath: File to save parameters
        """
        f = open(filepath, 'wb')
        pickle.dump(self.start_grid2world, f)
        pickle.dump(self.target_grid2world, f)
        pickle.dump(self.registration_methods, f)
        f.close()

    def load_warp_fields(self, filepath):
        """
        Load optimized warp fields and affine matrices from file.
        :param filepath: File of saved parameters
        """
        f = open(filepath, 'rb')
        self.start_grid2world = pickle.load(f)
        self.target_grid2world = pickle.load(f)
        self.registration_methods = pickle.load(f)
        f.close()

    def validate_registration_conditions(self, input_grid2world, is_inverse=False):
        """
        Making sure the input image for the transformaion fits the optimized voxel-to-space transformation
        :param input_grid2world: Voxel-to-space transformation of the input image
        :param is_inverse: Flag for switching transformation directions, 
                           If True, Check conditions for inverse transformation
        """
        assert self.registration_methods is not None, 'Registration needs to be optimized before application'
        if is_inverse:
            required_grid2world = self.target_grid2world
        else:
            required_grid2world = self.start_grid2world
        assert (input_grid2world == required_grid2world).all(), 'Input image grid is\n{}\nneeds to be\n{}'.format(
            input_grid2world, required_grid2world)

    def print_status(self, message):
        """
        Printing status reports according to verbosity
        :param message: Message to be printed in case verbosity is enabled
        """
        if self.verbose:
            print(message)
