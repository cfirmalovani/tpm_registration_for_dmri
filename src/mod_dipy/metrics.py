""" 
Original code of metrics for Symmetric Diffeomorphic Registration from DIPY
Includes variations for handling TPM based registration
"""

from __future__ import print_function
import abc
import numpy as np
from numpy import gradient
from scipy import ndimage
from dipy.align import vector_fields as vfu

from sys import platform

if platform == 'win32': 
    from mod_dipy import multimodal_crosscorr_win as mcc
elif platform == 'darwin':
    from mod_dipy import multimodal_crosscorr_mac as mcc
elif platform.startswith('linux'): # Using startswith in case of older python versions 
    from mod_dipy import multimodal_crosscorr_linux as mcc


class SimilarityMetric(object, metaclass=abc.ABCMeta):
    def __init__(self, dim):
        r""" Similarity Metric abstract class

        A similarity metric is in charge of keeping track of the numerical
        value of the similarity (or distance) between the two given images. It
        also computes the update field for the forward and inverse displacement
        fields to be used in a gradient-based optimization algorithm. Note that
        this metric does not depend on any transformation (affine or
        non-linear) so it assumes the static and moving images are already
        warped

        Parameters
        ----------
        dim : int (either 2 or 3)
            the dimension of the image domain
        """
        self.dim = dim
        self.levels_above = None
        self.levels_below = None

        self.static_image = None
        self.static_affine = None
        self.static_spacing = None
        self.static_direction = None

        self.moving_image = None
        self.moving_affine = None
        self.moving_spacing = None
        self.moving_direction = None
        self.mask0 = False

    def set_levels_below(self, levels):
        r"""Informs the metric how many pyramid levels are below the current one

        Informs this metric the number of pyramid levels below the current one.
        The metric may change its behavior (e.g. number of inner iterations)
        accordingly

        Parameters
        ----------
        levels : int
            the number of levels below the current Gaussian Pyramid level
        """
        self.levels_below = levels

    def set_levels_above(self, levels):
        r"""Informs the metric how many pyramid levels are above the current one

        Informs this metric the number of pyramid levels above the current one.
        The metric may change its behavior (e.g. number of inner iterations)
        accordingly

        Parameters
        ----------
        levels : int
            the number of levels above the current Gaussian Pyramid level
        """
        self.levels_above = levels

    def set_static_image(self, static_image, static_affine, static_spacing,
                         static_direction):
        r"""Sets the static image being compared against the moving one.

        Sets the static image. The default behavior (of this abstract class) is
        simply to assign the reference to an attribute, but
        generalizations of the metric may need to perform other operations

        Parameters
        ----------
        static_image : array, shape (R, C) or (S, R, C)
            the static image
        """
        self.static_image = static_image
        self.static_affine = static_affine
        self.static_spacing = static_spacing
        self.static_direction = static_direction

    def use_static_image_dynamics(self, original_static_image, transformation):
        r"""This is called by the optimizer just after setting the static image.

        This method allows the metric to compute any useful
        information from knowing how the current static image was generated
        (as the transformation of an original static image). This method is
        called by the optimizer just after it sets the static image.
        Transformation will be an instance of DiffeomorficMap or None
        if the original_static_image equals self.moving_image.

        Parameters
        ----------
        original_static_image : array, shape (R, C) or (S, R, C)
            original image from which the current static image was generated
        transformation : DiffeomorphicMap object
            the transformation that was applied to original image to generate
            the current static image
        """
        pass

    def set_moving_image(self, moving_image, moving_affine, moving_spacing,
                         moving_direction):
        r"""Sets the moving image being compared against the static one.

        Sets the moving image. The default behavior (of this abstract class) is
        simply to assign the reference to an attribute, but
        generalizations of the metric may need to perform other operations

        Parameters
        ----------
        moving_image : array, shape (R, C) or (S, R, C)
            the moving image
        """
        self.moving_image = moving_image
        self.moving_affine = moving_affine
        self.moving_spacing = moving_spacing
        self.moving_direction = moving_direction

    def use_moving_image_dynamics(self, original_moving_image, transformation):
        r"""This is called by the optimizer just after setting the moving image

        This method allows the metric to compute any useful
        information from knowing how the current static image was generated
        (as the transformation of an original static image). This method is
        called by the optimizer just after it sets the static image.
        Transformation will be an instance of DiffeomorficMap or None if
        the original_moving_image equals self.moving_image.

        Parameters
        ----------
        original_moving_image : array, shape (R, C) or (S, R, C)
            original image from which the current moving image was generated
        transformation : DiffeomorphicMap object
            the transformation that was applied to original image to generate
            the current moving image
        """
        pass

    @abc.abstractmethod
    def initialize_iteration(self):
        r"""Prepares the metric to compute one displacement field iteration.

        This method will be called before any compute_forward or
        compute_backward call, this allows the Metric to pre-compute any useful
        information for speeding up the update computations. This
        initialization was needed in ANTS because the updates are called once
        per voxel. In Python this is unpractical, though.
        """

    @abc.abstractmethod
    def free_iteration(self):
        r"""Releases the resources no longer needed by the metric

        This method is called by the RegistrationOptimizer after the required
        iterations have been computed (forward and / or backward) so that the
        SimilarityMetric can safely delete any data it computed as part of the
        initialization
        """

    @abc.abstractmethod
    def compute_forward(self):
        r"""Computes one step bringing the reference image towards the static.

        Computes the forward update field to register the moving image towards
        the static image in a gradient-based optimization algorithm
        """

    @abc.abstractmethod
    def compute_backward(self):
        r"""Computes one step bringing the static image towards the moving.

        Computes the backward update field to register the static image towards
        the moving image in a gradient-based optimization algorithm
        """

    @abc.abstractmethod
    def get_energy(self):
        r"""Numerical value assigned by this metric to the current image pair

        Must return the numeric value of the similarity between the given
        static and moving images
        """

class MCC_Metric(SimilarityMetric):

    def __init__(self, dim, sigma_diff=2.0, radius=4):
        r"""Multimodal Cross-Correlation Similarity metric.

        Parameters
        ----------
        dim : int (either 2 or 3)
            the dimension of the image domain
        sigma_diff : the standard deviation of the Gaussian smoothing kernel to
            be applied to the update field at each iteration
        radius : int
            the radius of the squared (cubic) neighborhood at each voxel to be
            considered to compute the cross correlation
        """
        super(MCC_Metric, self).__init__(dim)
        self.sigma_diff = sigma_diff
        self.radius = radius
        self._connect_functions()

    def _connect_functions(self):
        r"""Assign the methods to be called according to the image dimension

        Assigns the appropriate functions to be called for precomputing the
        cross-correlation factors according to the dimension of the input
        images
        """
        if self.dim == 3:
            self.precompute_factors = mcc.precompute_cc_factors_3d
            self.compute_forward_step = mcc.compute_cc_forward_step_3d
            self.compute_backward_step = mcc.compute_cc_backward_step_3d
            self.reorient_vector_field = vfu.reorient_vector_field_3d           
        else:
            raise ValueError('CCVI Metric not defined for dim. %d' % (self.dim))

    
    def probmap_gradient(self, image):
        r"""
        Calculate per modality gradient of multimodal image
        """
        gradient_map = np.zeros(image.shape + (3,))
        for idx in range(image.shape[-1]):
            for i, grad in enumerate(gradient(image[:,:,:,idx])):
                gradient_map[:,:,:,idx,i] = grad
        return gradient_map

    def initialize_iteration(self):
        r"""Prepares the metric to compute one displacement field iteration.

        Pre-computes the cross-correlation factors for efficient computation
        of the gradient of the Cross Correlation w.r.t. the displacement field.
        It also pre-computes the image gradients in the physical space by
        re-orienting the gradients in the voxel space using the corresponding
        affine transformations.
        """

        def invalid_image_size(image):
            min_size = self.radius * 2 + 1
            return any([size < min_size for size in image.shape[:-1]])

        msg = ("Each image dimension should be superior to 2 * radius + 1."
               "Decrease MCC_Metric radius or increase your image size")

        if invalid_image_size(self.static_image):
            raise ValueError("Static image size is too small. " + msg)
        if invalid_image_size(self.moving_image):
            raise ValueError("Moving image size is too small. " + msg)

        self.factors = self.precompute_factors(self.static_image,
                                               self.moving_image,
                                               self.radius)
        self.factors = np.array(self.factors)
        
        self.gradient_moving = self.probmap_gradient(self.moving_image)

        # Convert moving image's gradient field from voxel to physical space
        if self.moving_spacing is not None:
            for idx in range(self.gradient_moving.shape[-2]):
                self.gradient_moving[:,:,:,idx,:] /= self.moving_spacing
        if self.moving_direction is not None:
            for idx in range(self.gradient_moving.shape[-2]):   
                self.reorient_vector_field(self.gradient_moving[:,:,:,idx,:],
                                           self.moving_direction)

        self.gradient_static = self.probmap_gradient(self.static_image)

        # Convert moving image's gradient field from voxel to physical space
        if self.static_spacing is not None:
            for idx in range(self.gradient_static.shape[-2]):
                self.gradient_static[:,:,:,idx,:] /= self.static_spacing
        if self.static_direction is not None:
            for idx in range(self.gradient_static.shape[-2]):   
                self.reorient_vector_field(self.gradient_static[:,:,:,idx,:],
                                           self.static_direction)


    def free_iteration(self):
        r"""Frees the resources allocated during initialization
        """
        del self.factors
        del self.gradient_moving
        del self.gradient_static

    def compute_forward(self):
        r"""Computes one step bringing the moving image towards the static.

        Computes the update displacement field to be used for registration of
        the moving image towards the static image
        """
        displacement, self.energy = self.compute_forward_step(
            self.gradient_static, self.factors, self.radius)
        displacement = np.array(displacement)
        if np.sum(np.isnan(displacement)) > 0:
            print('NAN FOUND IN FOREWARD')
        #displacement[np.isnan(displacement)] = 0
        for i in range(self.dim):
            displacement[..., i] = ndimage.filters.gaussian_filter(
                displacement[..., i], self.sigma_diff)
        return displacement

    def compute_backward(self):
        r"""Computes one step bringing the static image towards the moving.

        Computes the update displacement field to be used for registration of
        the static image towards the moving image
        """
        displacement, energy = self.compute_backward_step(self.gradient_moving,
                                                          self.factors,
                                                          self.radius)
        displacement = np.array(displacement)
        if np.sum(np.isnan(displacement)) > 0:
            print('NAN FOUND IN BACKWARD')
        for i in range(self.dim - 1): 
            displacement[..., i] = ndimage.filters.gaussian_filter(
                displacement[..., i], self.sigma_diff)
        return displacement

    def get_energy(self):
        r"""Numerical value assigned by this metric to the current image pair

        Returns the Cross Correlation (data term) energy computed at the
        largest iteration
        """
        return self.energy
