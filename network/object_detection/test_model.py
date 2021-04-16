import cv2
import distutils.util
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
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from shutil import copyfile

from references.engine import train_one_epoch, evaluate
import references.utils as utils
import references.transforms as T

from box_hndlr import iso_package, get_box, draw_box
from bbox import PackageDataset, get_transform, show_pred_box

def main():
    model_name = sys.argv[1]   # should be .pt file in ./
    test_im_dir = sys.argv[2]  # assume test_im_dir/Normal and test_im_dir/Segmentation
    package_dir = bool(distutils.util.strtobool(sys.argv[3])) if len(sys.argv) > 3 else True
    output_iou = bool(distutils.util.strtobool(sy.argv[4])) if len(sys.argv) > 4 else False

    """
    If directory contains images of packages (package_dir == True), then new directory test_im_dir/NormalEval
    is made, containing the original images with the bounding box drawn on.

    If the directory contains images without packages (package_dir == False), then the new directory
    test_im_dir/SegmentationEval is made. This new directory contains images with the bounding box drawn on,
    only for the images in which a bounding box was predicted. 

    For both new directories, the name of the images in the new directory is the name of the original image 
    with 'eval' appended. Additionally, the new directories will contain a txt file containing information
    about the box corner coordinates, the box center coordinates, the area of the box, and the confidence
    level of the prediction for all the images with predictions made, as well as a summary of the evaluation.
    """

    # loading model
    loaded_model = torch.load(model_name)
    loaded_model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    assert os.path.isdir(os.path.join(test_im_dir, 'Normal'))
    num_test_images = len(os.listdir(os.path.join(test_im_dir, 'Normal')))

    os.mkdir(os.path.join(test_im_dir, 'NormalEval'))
    os.mkdir(os.path.join(test_im_dir, 'NormalEval', 'images'))    

    total_area = 0
    total_conf = 0
    # total_preds = len(os.listdir(os.path.join(test_im_dir, 'Normal')))
    total_preds = 0
    eval_data = {}

    
    for f in sorted(os.listdir(os.path.join(test_im_dir, 'Normal'))):
        f_im_path = os.path.join(test_im_dir, 'Normal', f)
        f_im = PIL.Image.open(f_im_path).convert("RGB")
        f_tensor = transforms.ToTensor()(f_im)

        with torch.no_grad():
            # making prediction
            mask_pred = loaded_model([f_tensor.to(device)])

        # package detected
        try:
            box_pred = mask_pred[0]['boxes'][0]
            conf_score = mask_pred[0]['scores'][0]

            # saving new image
            f_im_eval = show_pred_box(f_im_path, box_pred)
            new_f_name = f[:-4] + "_eval.png"
            f_im_eval.save(os.path.join(test_im_dir, 'NormalEval', 'images', new_f_name))

            # recording statistics
            x0, y0, x1, y1 = [int(i) for i in box_pred.tolist()]
            center_coords = ((x0 + x1) / 2, (y0 + y1) / 2)
            area = (x1 - x0) * (y1 - y0)

            stat_dict = {"box_coords": ((x0, y0), (x1, y1)),
                        "center_coords": center_coords,
                        "area": area,
                        "confidence": conf_score}

            total_area += area
            total_conf += conf_score
            total_preds += 1
            eval_data[new_f_name] =  stat_dict

        # package not detected
        except IndexError: 
            pass

    
    # writing statistics
    avg_area = total_area / total_preds
    avg_conf = total_conf / total_preds
    with open(os.path.join(test_im_dir, 'NormalEval', 'preds.txt'), 'w') as out_file:
        out_file.write("Summary\n")
        out_file.write(f"\tTotal packages detected: {total_preds}\n")
        out_file.write(f"\tRatio of images with packages detected: {total_preds/num_test_images}\n")
        out_file.write(f"\tAverage bbox area: {avg_area}\n")
        out_file.write(f"\tAverage prediction confidence: {avg_conf}\n")
        out_file.write(f"file_name,box_coords,center_coords,area,confidence\n")
        for f_name in eval_data.keys():
            sub_dict = eval_data[f_name]
            out_file.write(f"{f_name},{sub_dict['box_coords']},{sub_dict['center_coords']},{sub_dict['area']},{sub_dict['confidence']}\n")

        
    if output_iou:
        # creating dataloader
        test_dataset = PackageDataset(test_im_dir, get_transform(train=False))
        data_loader = torch.utils.data.DataLoader(
                        test_dataset, batch_size=2, shuffle=False, num_workers=4,
                        collate_fn=utils.collate_fn)

        # outputing IoU metrics
        evaluate(loaded_model, data_loader, device=device)
    

if __name__ == "__main__":
    main()