import argparse
import nibabel as nib
import numpy as np
import os

from tissue_classifier import Tissue_Classifier
from tpm_registration import TPM_Registration


class TPM_Registration_Pipeline:
    """
    Implementation of the TPM based registration pipeline for DTI images using TPM prediction
    Current pipeline supports pre-calculated, skull-stripped DTI images. 
    """

    def __init__(self):
        """
        Setting default values for the parser
        """
        self.abs_dir = os.path.dirname(os.path.abspath(__file__))
        self.dti_parameters = ['FA', 'L1', 'L2', 'L3']
        self.ref_tpm = os.path.join(self.abs_dir, '../data/mni_atlas_2mm.nii.gz')
        self.gaussian_window_size = 5
        self.gaussian_sigma = 1
        self.classifier_model_file = os.path.join(self.abs_dir, '../models/dti_tissue_classifier.sav')
        self.classifier_stats_file = os.path.join(self.abs_dir, '../models/dti_statistics.p')

    def create_parser(self):
        """
        Creating the parser
        """
        self.parser = argparse.ArgumentParser(prog='TPM Registration',
                                              description='Registration of DTI images \
                                                        using tissue probability maps')

        self.parser.add_argument("-i", "--input", type=str, metavar='FA_filename',
                                 help="filename of input FA map - must be in the same directory with Lambda maps",
                                 required=True)
        self.parser.add_argument("-r", "--ref", "--tpm", type=str, metavar='ref_TPM_filename',
                                 help="reference for TPMs of the target space (default is MNI)")
        self.parser.add_argument("-o", "--out", type=str, metavar='warped_filename',
                                 help="destination folder for output (warped) DTI maps (default is source directory)")
        self.parser.add_argument("-ot", "--out_tpm", type=str, nargs='?', default=None, const='default',
                                 metavar='TPM_savedir',
                                 help="if activated, saving calculated TPM in destination folder "
                                      "(default is source directory)")
        self.parser.add_argument("-m", "--mask", type=str, metavar='mask_filename',
                                 help="binary mask file (default will assume the DTI maps are already masked)")
        self.parser.add_argument("--interpolation", "--interp", type=str, metavar='interpolation_method',
                                 default='linear',
                                 help="interpolation method for transformation: linear (default) / nearest")
        self.parser.add_argument("-sw", "--save_wrap", type=str, nargs='?', default=None, const='default',
                                 metavar='warp_savedir',
                                 help="if activated, saving warp fields in destination folder "
                                      "(default is source directory)")
        self.parser.add_argument("--smoothing", type=int, nargs='*', metavar='parameter',
                                 help="if activated, gaussian smoothing parameters for TPM - "
                                      "may insert values for window size and sigma (default are 5, 1)")
        self.parser.add_argument("-v", "--verbose", action="store_true",
                                 help="if activated, provide status reports of the pipeline")

    def fsl_dti_file_reader(self, FA_filepath):
        """
        Reads the DTI files and create the image for the tissue classifier
        :param FA_filepath: file path of one the FA image, received from the parser
        :return dti_image: The created 4D array of the needed DTI parameters
        :return dti_affine: Voxel-to-space transformation of the DTI image
        :return dirpath: Directory of the given file, to be used as default folder
        :return basename: Name prefix of the file, to be used as prefix for the output as well
        """
        dirpath, filename = os.path.split(FA_filepath)
        filename_pure = filename[:filename.find('.nii')]
        FA_index = filename_pure.find('FA')
        basename = [filename_pure[:FA_index], filename_pure[FA_index + 2:]]
        # basename = filename.split('.')[0].split('_')[0]

        self.print_status('Loading FA file - {}'.format(os.path.basename(FA_filepath)))
        FA_scan = nib.load(FA_filepath)
        dti_affine = FA_scan.affine
        dti_image = FA_scan.get_fdata()[..., np.newaxis]
        for l_idx in range(1, 4):
            L_filepath = os.path.join(dirpath, '{}L{}{}.nii.gz'.format(basename[0], l_idx, basename[1]))
            self.print_status('Loading L{} file - {}'.format(l_idx, os.path.basename(L_filepath)))
            assert os.path.exists(L_filepath), '{} file not found'.format(os.path.basename(L_filepath))
            L_image = nib.load(L_filepath).get_fdata()
            dti_image = np.concatenate((dti_image, L_image[..., np.newaxis]), axis=-1)

        return dti_image, dti_affine, dirpath, basename

    def parse_input(self, test_args=None):
        """
        Parsing given inputs to different program variables
        :param test_args: Test input commands to the parser (replacing stdin input)
        """
        if test_args is None:
            args = self.parser.parse_args()
        else:
            args = self.parser.parse_args(test_args)

        self.verbose = args.verbose
        self.interpolation = args.interpolation

        if args.smoothing is not None and len(args.smoothing) not in (0, 2):
            self.parser.error('Either give no values for action, or two, not {}.'.format(len(args.smoothing)))

        dti_image, dti_affine, default_dirpath, self.basename = self.fsl_dti_file_reader(args.input)

        if args.mask is not None:
            mask_image = nib.load(args.mask)
            assert (mask_image.affine == dti_affine).all(), 'Mask scan does not match input scan!'
            mask_image = mask_image.get_fdata()
        else:  # Assume DTI image is already masked, calculate from non-zero elements in matrix
            mask_image = 1. * (np.sum(dti_image != 0, axis=-1) > 0)

        if args.out is None:
            self.out_warped_dir = default_dirpath
        else:
            self.out_warped_dir = args.out

        if args.ref is not None:
            self.ref_tpm = args.ref

        if args.out_tpm is None:
            self.tpm_output = None
        elif args.out_tpm == 'default':
            self.tpm_output = os.path.join(default_dirpath, '{}TPM{}.nii.gz'.format(self.basename[0], self.basename[1]))
        else:
            self.tpm_output = os.path.join(args.out_tpm, '{}TPM{}.nii.gz'.format(self.basename[0], self.basename[1]))

        if args.save_wrap is None:
            self.save_warpfields = None
        elif args.save_wrap == 'default':
            self.save_warpfields = os.path.join(default_dirpath,
                                                '{}Warpfields{}.sav'.format(self.basename[0], self.basename[1]))
        else:
            self.save_warpfields = os.path.join(args.save_wrap,
                                                '{}Warpfields{}.sav'.format(self.basename[0], self.basename[1]))

        if args.smoothing is None:
            self.smooth_tpm = False
        elif len(args.smoothing) == 0:
            self.smooth_tpm = True
        elif len(args.smoothing) == 2:
            self.gaussian_window_size = args.smoothing[0]
            self.gaussian_sigma = args.smoothing[1]

        return dti_image, mask_image, dti_affine

    def run_pipeline(self, test_args=None):
        """
        Running the TPM registration pipeline according to parser's inputs
		:param test_args: Possible test input for the parser from within the python program
        """
        self.create_parser()
        dti_image, mask_image, dti_affine = self.parse_input(test_args)

        # Tissue classification
        classifier = Tissue_Classifier(self.classifier_model_file, self.classifier_stats_file, verbose=self.verbose)
        tpm = classifier.predict_tpm(dti_image, mask_image, self.smooth_tpm,
                                     self.gaussian_window_size, self.gaussian_sigma)
        if self.tpm_output is not None:
            self.print_status('Saving TPMs to file {}'.format(os.path.basename(self.tpm_output)))
            nib_scan = nib.Nifti1Image(tpm, dti_affine)
            nib.save(nib_scan, self.tpm_output)

        # TPM registration
        tpm_registration = TPM_Registration(verbose=self.verbose, interpolation=self.interpolation)
        target_scan = nib.load(self.ref_tpm)
        target_affine = target_scan.affine
        target_image = target_scan.get_fdata()
        tpm_registration.optimize_tpm_registration(tpm, dti_affine, target_image, target_affine)

        self.print_status('Saving warped DTI images')
        warped_dti_image = tpm_registration.register_scan(dti_image, dti_affine)
        for idx, dti_type in enumerate(self.dti_parameters):
            nib_scan = nib.Nifti1Image(warped_dti_image[..., idx], target_affine)
            output_file = os.path.join(self.out_warped_dir,
                                       'warped_{}{}{}.nii.gz'.format(self.basename[0], dti_type, self.basename[1]))
            nib.save(nib_scan, output_file)

        if self.tpm_output is not None:
            warped_tpm_file = 'warped_' + os.path.basename(self.tpm_output)
            self.print_status('Saving warped TPMs to file {}'.format(warped_tpm_file))
            warped_tpm = tpm_registration.register_scan(tpm, dti_affine, is_tpm=True)
            nib_scan = nib.Nifti1Image(warped_tpm, target_affine)
            nib.save(nib_scan, os.path.join(os.path.dirname(self.tpm_output), warped_tpm_file))

        if self.save_warpfields is not None:
            self.print_status('Saving warp fields to file {}'.format(os.path.basename(self.save_warpfields)))
            tpm_registration.save_warp_fields(self.save_warpfields)

    def print_status(self, message):
        """
        Printing status reports according to verbosity
        :param message: Message to be printed in case verbosity is enabled
        """
        if self.verbose:
            print(message)


if __name__ == '__main__':
    tpm_pipeline = TPM_Registration_Pipeline()
    tpm_pipeline.run_pipeline()
