import nibabel as nib
from nilearn import datasets, image

# 1) Standard GM mask (downloaded automatically to nilearn_data/)
gm = datasets.load_mni152_gm_mask()

# 2) Resample to your exact HCP-YA preproc grid (2mm MNI, same affine/shape)
ref = nib.load("/path/to/fmri/derivatives/sub-211114/func/sub-211114_task-rest_dir-LR_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
ref_3d = nib.Nifti1Image(ref.get_fdata()[..., 0], ref.affine)
gm_res = image.resample_to_img(gm, ref_3d, interpolation="nearest")

out = "/path/to/user/data/masks/mni152_gm_mask_res2_hcpya.nii.gz"
nib.save(gm_res, out)
print("saved:", out, "shape:", gm_res.shape)