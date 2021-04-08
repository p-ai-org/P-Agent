import cv2
import numpy as np
import PIL
from PIL import Image
from pypfm import PFMLoader

"""
Helper functions for box stuff
"""


loader = PFMLoader(compress=False)


def iso_package(pfm_file, rgb_vals):
    """
    Function that returns np.array of the isolated segmentation of the package
    """
    assert isinstance(rgb_vals, np.ndarray)
    assert rgb_vals.shape == (3,)

    pfm_loaded = loader.load_pfm(pfm_file)
    pfm_arr = ((np.asarray(pfm_loaded)*255)).astype(np.uint8)
    package_np = np.where(pfm_arr == rgb_vals, rgb_vals, np.zeros((3,))).astype(np.uint8)

    return np.flipud(package_np)  # the box is upside down, need to flip


def get_box(seg_arr):
    """
    Function that returns the coordinates corresponding to corners of bounding box
    seg_arr: numpy array of segmenatation (i.e. output of iso_package)
    """
    xy_indices = np.where(np.add.reduce(seg_arr, axis=2))
    x_0 = np.min(xy_indices[1])
    y_0 = np.min(xy_indices[0])
    x_1 = np.max(xy_indices[1])
    y_1 = np.max(xy_indices[0])

    return [x_0, y_0, x_1, y_1]


def draw_box(img_path, seg_path, rgb_vals):
    """
    Function to generate image with bounding box from segmentation label
    """
    seg_np = iso_package(seg_path, rgb_vals)
    x0, y0, x1, y1 = get_box(seg_np)
    cv_img = cv2.imread(img_path)
    cv_arr = cv2.rectangle(cv_img, (x0, y0), (x1, y1), (255, 0, 255), thickness=2)
    cv_arr = cv2.cvtColor(cv_arr, cv2.COLOR_BGR2RGB)

    return PIL.Image.fromarray(cv_arr)
