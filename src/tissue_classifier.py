"""
XGBoost based DTI tissue classifier
             
@author: Cfir Malovani
"""

import pickle
import numpy as np
from scipy.ndimage.filters import convolve


class Tissue_Classifier:
    """
    XGboost based DTI tissue classifier.
    Prediction output is a 4D array, with the last dimension contains
    the different TPMs: GM, WM, CSF, Background
    """
    def __init__(self, model_file, stats_file):
        """
        Tissue classifier initializtion
        :param model_file: Pre-trained version of the classifier. Saved using the pickle module
        :param stat_file: Normalization paramaters for the different DTI images. Saved using the pickle module
        """
        # Load classifier files
        f = open(model_file, 'rb')
        self.model = pickle.load(f)
        f.close()
      
        f = open(stats_file, 'rb')
        self.dti_statistics = pickle.load(f)
        f.close()
        
        # Classifier is dedicated for DTI parameters
        self.parameters_list = ['FA', 'L1', 'L2', 'L3']
        self.num_parameters = len(self.parameters_list)
        
        
    def validate_inputs(self, input_dti, input_mask):
        """
        Validation of input scan and mask
        :param input_dti: Input DTI image
        :param input_mask: Input mask image
        """
        assert (input_dti.shape[-1] == self.num_parameters), 'Input scan should be of shape (..., {})'.format(self.num_parameters)
        assert input_dti.shape[:-1] == input_mask.shape, 'Scan and mask dimensions do not agree'

        
    def dti_normalization(self, input_dti, input_mask):
        """
        Normalization of DTI image by pre-calculated mean and STD from the loaded stats_file
        :param input_dti: Input DTI image
        :param input_mask: Input mask image
        :return output_dti: Normalized and masked DTI image
        """
        self.validate_inputs(input_dti, input_mask)

        output_dti = np.zeros(input_dti.shape)
        for idx in range(self.num_parameters):
            current_type = self.parameters_list[idx]
            current_mean = self.dti_statistics[current_type]['mean']
            current_std = self.dti_statistics[current_type]['std']
            output_dti[..., idx] = input_mask * ((input_dti[..., idx] - current_mean) / current_std)
    
        return output_dti

    def predict_tpm(self, input_dti, input_mask, smooth_tpm = False, window_size=5, sigma=1):
        """
        Prediction of tissue segmentation from the DTI image
        :param input_dti: Input DTI image
        :param input_mask: Input mask image
        :param smooth_tpm: Flag for TPM smoothing
        :param window_size: Size of gaussian filter window (if smooth_tpm = True)
        :param sigma: Width of gaussian (if smooth_tpm = True)
        :return output_tpm: 4D array of TPMs
        """
        self.validate_inputs(input_dti, input_mask)
                    
        input_shape = input_mask.shape
        output_tpm = np.zeros(input_shape + (4,))
        input_dti = self.dti_normalization(input_dti, input_mask)
    
        label_indices = np.where(input_mask > 0)
        dti_features = np.zeros((len(label_indices[0]), input_dti.shape[-1]))
        for idx in range(input_dti.shape[-1]):
            single_parameter = input_dti[..., idx]
            dti_features[:,idx] = single_parameter[label_indices]
            
        outputs_predictions = self.model.predict_proba(dti_features)
        for idx in range(3):
            single_tpm = np.zeros(input_shape)
            single_tpm[label_indices] = outputs_predictions[:, idx]
            output_tpm[..., idx] = single_tpm
        
        output_tpm[..., 0:3] = np.roll(output_tpm[...,0:3], -1, axis = -1)
        output_tpm[..., -1] = 1 - input_mask
        
        ## Apply smoothing (if chosen to)
        if smooth_tpm:
            output_tpm = self.smooth_tpm(output_tpm, window_size, sigma)
        
        return output_tpm


    def create_3d_gaussian_filter(self, window_size=5, sigma=1):
        """
        Creation of 3D gaussian for TPM smoothing
        :param window_size: Size of filter window
        :param sigma: Width of gaussian
        :return gaussian_grid: 3D gaussian filter
        """
        x_vector = np.arange(window_size) - window_size // 2
        x_matrix, y_matrix, z_matrix = np.meshgrid(x_vector, x_vector, x_vector)
        gaussian_grid = np.exp(-(x_matrix ** 2 + y_matrix ** 2 + z_matrix ** 2) / (2 * sigma ** 2))
        gaussian_grid /= np.sum(gaussian_grid)
        return gaussian_grid


    def smooth_tpm(self, input_tpm, window_size=5, sigma=1):
        """
        TPM smoothing using gaussian filter
        :param input_tpm: TPM for smoothing
        :param window_size: Size of gaussian filter window
        :param sigma: Width of gaussian
        :return smoothed_tpm: TPM after gaussian smoothing
        """
        input_mask = 1 - input_tpm[..., -1]
        
        # Smoothing TPMs
        smoothed_tpm = np.zeros(input_tpm.shape)
        gaussian_filter = self.create_3d_gaussian_filter(window_size, sigma)
        for idx in range(input_tpm.shape[-1] - 1):
            smoothed_tpm[..., idx] = convolve(input_tpm[..., idx], gaussian_filter)
        smoothed_tpm[..., :-1] *= np.repeat(input_mask[..., np.newaxis], 3, axis=-1)
        smoothed_tpm[..., -1] = 1 - input_mask
    
        # Renormalizing the TPMs
        normal_factor = np.repeat(np.expand_dims(np.sum(smoothed_tpm, axis=-1), axis=-1), smoothed_tpm.shape[-1],
                                  axis=-1)
        smoothed_tpm /= normal_factor
    
        return smoothed_tpm