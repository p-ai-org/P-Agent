import os
import pdb

import numpy as np
import pickle
from PIL import Image

def pickle_images(data_path=os.path.join("..", "data", "package_predict")):
    """
    Reads images and labels stored in data_path and saves as pickle file.

    Expected directory structure:
    - data_path
        - folder_1
            - airsim_rec.txt (labels)
            - images
                - 1-1.png
                - 1.2.png
                - ...
        - folder_2
            - ...
        - ...

    Assumptions:
    - Labels are sorted alphabetically by image file name
    - No other files are in "images" folders
    """
    
    all_images = None
    all_labels = None

    for folder in [f.path for f in os.scandir(data_path) if f.is_dir()]:
        print(folder)

        labels = np.genfromtxt(os.path.join(folder, "airsim_rec.txt"), delimiter="\t")
        labels = labels[1:, 1:-1].astype("float64")

        if all_labels is None:
            all_labels = labels
        else:
            all_labels = np.append(all_labels, labels, axis=0)

        for image_f in [f.path for f in sorted(os.scandir(os.path.join(folder, "images")),
                key=lambda e: e.name)]:
            assert image_f.split(".")[-1] == "png"

            image_np = np.asarray(Image.open(image_f)).astype("float64")[:, :, :3] / 255

            ### Transformation used to flatten images if desired, row-major
            # image_np = np.transpose(image_np, (2, 0, 1)).flatten()

            image_np = np.expand_dims(image_np, axis=0)

            if all_images is None:
                all_images = image_np
            else:
                all_images = np.append(all_images, image_np, axis=0)

        assert all_labels.shape[0] == all_images.shape[0]

    obj = {"data": all_images, "labels": all_labels}
    pickle.dump(obj, open(os.path.join("..", "data", "package_predict", "images_labels.pkl"), "wb"))
    print("data/package_predict/images_labels.pkl saved")

if __name__ == "__main__":
    pickle_images()
