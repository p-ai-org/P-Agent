import cv2
import numpy as np
import os
import sys
import shutil
import zipfile
import py7zr
<<<<<<< HEAD
from box_hndlr import iso_package, get_box, draw_box
from pathlib import Path
from pypfm import PFMLoader
=======
# from box_hndlr import iso_package, get_box, draw_box
# from pypfm import PFMLoader
>>>>>>> 709b7c7177d4fc307a5b12faafc05d0cc4d083c1
from shutil import copyfile

"""
Script to split data into two directories, one consisting of images including the package and the other consisting of images without the package
"""

PACKAGE_COLOR = np.array([147, 0, 190])  # need to ensure this is always correct

def main():
    z_file = sys.argv[1]  # the name of the zip file with hierarchy z_file/Normal(Segmentation)
    new_dir = sys.argv[2]  # the name of the new dir with hierarchy new_dir/no_package(package)/Normal(Segmentation)

    path = Path(z_file)
    parent_path = path.parent.absolute()
    if z_file[-4:] == ".zip":
        with zipfile.ZipFile(z_file, 'r') as zip_ref:
            zip_ref.extractall(path=parent_path)
        z_file_base = z_file[:-4]
    elif z_file[-3:] == ".7z":
        with py7zr.SevenZipFile(z_file, 'r') as zip_ref:
            zip_ref.extractall(path=parent_path)
        z_file_base = z_file[:-3]

    old_names = {}
    for i, f in enumerate(sorted(os.listdir(os.path.join(z_file_base, 'Normal')))):
        fname = str(i).zfill(4)
        root = f[:-5]
        shutil.move(os.path.join(z_file_base, 'Normal', f), os.path.join(z_file_base, 'Normal', fname + '.png'))
        shutil.move(os.path.join(z_file_base, 'Segmentation', root + '1.pfm'), os.path.join(z_file_base, 'Segmentation', fname + '.pfm'))

    num_images = len(os.listdir(os.path.join(z_file_base, 'Normal')))
    assert num_images == len(os.listdir(os.path.join(z_file_base, 'Segmentation')))
    
    print(f"Number of files in directory: {num_images}")

    includes_package = []
    without_package = []

    for i, f in enumerate(sorted(os.listdir(os.path.join(z_file_base, 'Normal')))):
        img_path = os.path.join(z_file_base, 'Normal', f)
        seg_path = os.path.join(z_file_base, 'Segmentation', f[:-4] + '.pfm')

        try:
            out_im = draw_box(img_path, seg_path, PACKAGE_COLOR)
            # print(f'img: {f}')
            # display(out_im)
            includes_package.append(f)
        except ValueError:
            without_package.append(f)
            pass

    print(f"Number of images with package in frame: {len(includes_package)}")
    print(f"Number of images without package in frame: {num_images - len(includes_package)}")

    os.mkdir(os.path.join(z_file_base, new_dir))    
    os.mkdir(os.path.join(z_file_base, new_dir, 'w_package'))
    os.mkdir(os.path.join(z_file_base, new_dir, 'wo_package'))

    os.mkdir(os.path.join(z_file_base, new_dir, 'w_package', 'Normal'))
    os.mkdir(os.path.join(z_file_base, new_dir, 'w_package', 'Segmentation'))

    os.mkdir(os.path.join(z_file_base, new_dir, 'wo_package', 'Normal'))
    os.mkdir(os.path.join(z_file_base, new_dir, 'wo_package', 'Segmentation'))

    for i, fname in enumerate(includes_package):
        new_name = str(i).zfill(3) + ".png"
        copyfile(os.path.join(z_file_base, 'Normal', fname), os.path.join(z_file_base, new_dir, 'w_package', 'Normal', new_name))
        copyfile(os.path.join(z_file_base, 'Segmentation', fname[:-2] + 'fm'), os.path.join(z_file_base, new_dir, 'w_package', 'Segmentation', new_name[:-2] + 'fm'))

    for i, fname in enumerate(without_package):
        new_name = str(i).zfill(3) + ".png"
        copyfile(os.path.join(z_file_base, 'Normal', fname), os.path.join(z_file_base, new_dir, 'wo_package', 'Normal', new_name))
        copyfile(os.path.join(z_file_base, 'Segmentation', fname[:-2] + 'fm'), os.path.join(z_file_base, new_dir, 'wo_package', 'Segmentation', new_name[:-2] + 'fm'))


if __name__ == "__main__":
    main()
