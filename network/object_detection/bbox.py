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
import references.utils
import references.transforms as T

import clean_data
from box_hndlr import iso_package, get_box, draw_box

# sketchy workaround to avoid OMP: Error #15: Initializing libiomp5.dylib ... on MacOS
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

PACKAGE_COLOR = np.array([147, 0, 190])  # need to ensure this is always correct

class PackageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # loading and aligning image and segmentation files
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Normal"))))
        self.segs = list(sorted(os.listdir(os.path.join(root, "Segmentation"))))

    def __getitem__(self, idx):
        # load images and segmentations
        img_path = os.path.join(self.root, "Normal", self.imgs[idx])
        seg_path = os.path.join(self.root, "Segmentation", self.segs[idx])
        img = PIL.Image.open(img_path).convert("RGB")

        #TODO: un-hardcode the color
        seg = iso_package(seg_path, PACKAGE_COLOR)
        single_channel = seg.sum(axis=2)
        masks = np.where(single_channel, 1, 0)
        masks = torch.as_tensor([masks], dtype=torch.uint8)    

        # get box coordinates (note we are assuming max of one object)
        try:
            x0, y0, x1, y1 = get_box(seg)
            # ensuring we have positive height and width for box
            if x0 >= x1:
                x1 = x0 + 1
            if y0 >= y1:
                y1 = y0 + 1
        except ValueError:
            # print("no package in frame")
            # x0, y0, x1, y1 = [0]*4
            x0, y0, x1, y1 = 0, 0, 1, 1  # temporary fix to prevent non-positive box dimensions error

        # get area
        area = (x1 - x0) * (y1 - y0)

        # (we could add the coordinates of the segmentation if we wanted)
        target = {}
        target["boxes"] = torch.as_tensor([[x0, y0, x1, y1]], dtype=torch.float32)
        # target["image_id"] = torch.tensor([idx])
        # target["area"] = torch.tensor([area])
        # target["includes_package"] = torch.as_tensor(includes_package)
        target["labels"] = torch.tensor([1])
        target["masks"] = masks
        target["image_id"] = torch.tensor([idx])
        target["area"] = torch.tensor([area])
        target["iscrowd"] = torch.tensor([0])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    # loading instance segmenation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace pretrained head with new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # get number of input features for mask classfier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # replace mask predictor with new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        num_classes)

    return model


# TODO: add more transforms (i.e. scale, skew, rotate, etc.)
def get_transform(train):
    transforms = []
    # convert PIL image to PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # randomly flip training images during training
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# function for testing
def show_pred_box(test_im_path, boxes):
    """
    test_im_path: the name of the image file inputed for prediction
    boxes: something of the form `prediction[0]['boxes'][0]`
    """
    x0, y0, x1, y1 = [int(i) for i in boxes.tolist()]
    cv_img = cv2.imread(test_im_path)
    cv_arr = cv2.rectangle(cv_img, (x0, y0), (x1, y1), (255, 0, 255), thickness=2)
    cv_arr = cv2.cvtColor(cv_arr, cv2.COLOR_BGR2RGB)

    return PIL.Image.fromarray(cv_arr)


def testing_images(im_dir, model, show_im=False):
    for f in sorted(os.listdir(im_dir)):
        f_im = PIL.Image.open(os.path.join(im_dir, f)).convert("RGB")
        f_tensor = transforms.ToTensor()(f_im)
        with torch.no_grad():
            f_pred = model([f_tensor.to(device)])
        print(f"image {f}")
        try:
            print(f"bbox confidence: {f_mask_pred[0]['scores'][0]}")
            if show_im:
                display(show_pred_box(bad_file, f_mask_pred[0]['boxes'][0]))
        except IndexError:
            print("no package found")
            print(f"scores: {f_mask_pred[0]['scores']}")
            print(f"bboxes: {f_mask_pred[0]['boxes']}")
            if show_im:
                display(f_im)


def main():
    """
    For data_dir, we assume hierarchy (data_dir/Normal, data_dir/Segmentation)
    and all images include package
    model_out_path: the path/name of the model file saved
    """
    data_dir = sys.argv[1]
    model_out_path = sys.argv[2]
    num_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 6

    # using dataset and defined transformations
    dataset = PackageDataset(data_dir, get_transform(train=True))
    dataset_test = PackageDataset(data_dir, get_transform(train=False))

    # split dataset into train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-20])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-20:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=references.utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=references.utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # only background and package for this dataset
    num_classes = 2

    # get model
    my_model = get_model_instance_segmentation(num_classes)

    # move model to correct device
    my_model.to(device)

    # construct optimizer
    params = [p for p in my_model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # lr_scheduler that decreases learning rate by 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)    

    # TRAINING
    for epoch in range(num_epochs):
        # train for one epoch and printing every 10 iterations
        train_one_epoch(my_model, optimizer, data_loader, device, epoch, print_freq=10)
        # update learning rate
        lr_scheduler.step()
        # evaluate on test dataset
        evaluate(my_model, data_loader_test, device=device)

    torch.save(my_model, model_out_path)


if __name__ == "__main__":
    main()