import cv2
import numpy as np
import os
import PIL
import shutil
import sys
import torch
import torch.utils.data
import torchvision
import zipfile
from IPython.display import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from shutil import copyfile

from references.engine import train_one_epoch, evaluate
import references.utils as utils
import references.transforms as T

from box_hndlr import iso_package, get_box, draw_box
from bbox import PackageDataset, get_transform

def main():
    model_name = sys.argv[1]   # should be .pt file in ./
    test_im_dir = sys.argv[2]  # assume test_im_dir/Normal and test_im_dir/Segmentation

    """
    Planning to split test data into includes box and doesn't include package before evaluation
    As is, the metrics are thrown off by the loss of predictions where package isn't in the frame
    since we currently require the target to include a bbox even if there is no package
    -- evaluate normally for sub test set with packages in image
    -- let accuracy be % of predictions in which no package was correctly predicted for sub test set without packages
    """

    # loading model
    loaded_model = torch.load(model_name)
    loaded_model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # creating dataloader
    test_dataset = PackageDataset(test_im_dir, get_transform(train=False))
    data_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=2, shuffle=False, num_workers=4,
                    collate_fn=utils.collate_fn)

    evaluate(loaded_model, data_loader, device=device)
    

if __name__ == "__main__":
    main()