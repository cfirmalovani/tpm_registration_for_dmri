import argparse
import nibabel as nib
import numpy as np
import os

from tpm_registration import TPM_Registration

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TPM Registration Applywarp',
                                     description='Application of saved warp fields')

    parser.add_argument("-i", "--input", type=str, metavar='MRI scan to register',
                        help="A scan file to be registered",
                        required=True)
    parser.add_argument("-w", "--saved_warps", type=str, metavar='Saved warp fields',
                        help="Saved warped fields to be applied",
                        required=True)
    parser.add_argument("-o", "--out", type=str, metavar='Output path for warped scan',
                        help="destination folder for output (warped) DTI maps (default is source directory)")
    parser.add_argument("--inverse", action='store_true', help="Apply the inverse transformation on input")
    parser.add_argument("--interpolation", "--interp", type=str, metavar='interpolation_method',
                        default='linear',
                        help="interpolation method for transformation: linear (default) / nearest")
    args = parser.parse_args()

    if args.out is None:
        out_warped_dir, filename = os.path.split(args.input)
        header = 'inv_warped_' if args.inverse else 'warped_'
        out_warped_path = os.path.join(out_warped_dir, header + filename)
    else:
        out_warped_path = args.out

    tpm_registration = TPM_Registration(interpolation=args.interpolation)
    tpm_registration.load_warp_fields(args.saved_warps)
    nib_scan = nib.load(args.input)
    scan = nib_scan.get_fdata()
    scan_affine = nib_scan.affine

    if args.inverse:
        warped_scan = tpm_registration.inverse_register_scan(scan, scan_affine)
        nib_scan = nib.Nifti1Image(warped_scan, tpm_registration.start_grid2world)
        nib.save(nib_scan, out_warped_path)
    else:
        warped_scan = tpm_registration.register_scan(scan, scan_affine)
        nib_scan = nib.Nifti1Image(warped_scan, tpm_registration.target_grid2world)
        nib.save(nib_scan, out_warped_path)
