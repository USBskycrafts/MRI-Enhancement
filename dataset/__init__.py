from .cv.ImageFromMRI import ImageFromMRI
from .nifti.NIFTI1Loader import NIFTI1Loader

dataset_list = {
    "ImageFromMRI": ImageFromMRI,
    "NIFTI": NIFTI1Loader
}
