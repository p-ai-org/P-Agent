import numpy as np
import os
import sys
import shutil
import zipfile
from box_hndlr import iso_package, get_box, draw_box
from pypfm import PFMLoader
from shutil import copyfile

def main():
    z_file = sys.argv[1]
    new_dir = sys.argv[2]

    with zipfile.ZipFile(z_file, 'r') as zip_ref:
        zip_ref.extractall()

    z_file_base = z_file[:-4]
    old_names = {}
    for i, f in enumerate(sorted(os.listdir(os.listdir(z_file_base, 'Normal')))):
        fname = str(i).zfill(4)
        root = f[:-5]
        shutil.move(os.path.join(z_file_base, 'Normal'), os.path.join(z_file_base, 'Normal', fname + '.png'))
        shutil.move(os.path.join(z_file_base, 'Segmentation', root + '1.pfm'), os.path.join(z_file_base, 'Segmentation', fname + '.pfm'))

    num_images = len(os.listdir(os.path.join(z_file_base, 'Normal'))
    assert  num_images == len(os.listdir(os.path.join(z_file_base, 'Segmentation')))
    
    print(f"Number of files in directory: {num_images}")

    includes_package = []

    for i, f in enumerate(sorted(os.listdir(os.path.join(z_file_base, 'Normal')))):
        img_path = os.path.join(z_file_base, 'Normal', f)
        seg_path = os.path.join(z_file_base, 'Segmentation', f[:-4] + '.pfm')

        try:
            # out_im = draw_box(img_path, seg_path, PACKAGE_COLOR)
            # print(f'img: {f}')
            # display(out_im)
            includes_package.append(f)
        except ValueError:
            pass

    print(f"Number of images with package in frame: {len(includes_package)}")
    print(f"Number of images without package in frame: {num_images - len(includes_package)}")

    os.mkdir(os.path.join(new_dir, 'Normal'))
    os.mkdir(os.path.join(new_dir, 'Segmentation'))

    for i, fname in enumerate(includes_package):
        new_name = str(i).zfill(3) + ".png"
        copyfile(os.path.join(z_file_base, 'Normal', fname), os.path.join(new_dir, 'Normal', new_name))
        copyfile(os.path.join(z_file_base, 'Segmentation', fname[:-2] + 'fm'), os.path.join(new_dir, 'Segmentation', new_name[:-2] + 'fm'))


if __name__ == "__main__":
    main()
