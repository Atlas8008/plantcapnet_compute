import os
import sys; sys.path.append("../../.."); sys.path.append("../../../networks")
import torch
import argparse
import numpy as np
import torchmetrics

from torch import nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.augmentations import Preprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation of segmentations.")

    parser.add_argument("model_file", type=str, help="The path to the model file to evaluate.")

    #add_argparse_model_parameters(parser, include_lr=False)

    parser.add_argument("--source_folder", type=str, default="./eval_segmentations", help="The folder from which to extract images and segmentations.")
    parser.add_argument("--gpu", type=str, default="0", help="The GPU to select.")

    args = parser.parse_args()

class_names_bg = [
    "Background",
    "Irrelevance",
]
class_names_base = [
    "A. millefolium",
    "C. jacea",
    "Grasses",
    "L. corniculatus",
    "M. lupulina",
    "P. lanceolata",
    "S. autumnalis",
    "T. pratense",
]
class_names_nodl = class_names_base + class_names_bg
class_names = class_names_base + ["Dead Litter"] + class_names_bg


class SegmentationComparisonDataset(torch.utils.data.Dataset):
    def __init__(self, source_folder, image_transform):
        super().__init__()
        self.image_list = []
        self.segmentation_list = []
        self.image_transform = image_transform

        with open(os.path.join(source_folder, "images_and_segmentations.txt"), "r") as f:
            lines = f.read().strip().split("\n")
            lines = [l.split("\t") for l in lines]

            for t in lines:
                self.image_list.append(os.path.join(source_folder, "images", t[0]))
                self.segmentation_list.append(os.path.join(source_folder, "segmentations", t[1]))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        segmentation_path = self.segmentation_list[idx]

        image = Image.open(img_path).convert("RGB")
        segmentation = load_segmentations_from_file(segmentation_path)

        if self.image_transform:
            image = self.image_transform(image)

        return image, segmentation


def load_segmentations_from_file(fname):
    class_indices = np.array(Image.open(fname))[:, :, 0]

    return class_indices

def merge_dead_litter_into_background(tensor):
    dl_idx = class_names.index("Dead Litter")
    bg_idx = class_names.index("Background")

    tensor[tensor == dl_idx] = bg_idx

    # Reduce all indices after dead litter by 1
    tensor = torch.where(
        tensor >= dl_idx,
        tensor - 1,
        tensor
    )

    return tensor


def eval_miou(model, source_folder, device, image_normalization="torch", dead_litter_as_bg=False, augmented_eval=False, img_size_hw=(768, 1536)):
    image_size = img_size_hw

    # Transforms
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        Preprocess(image_normalization),
    ])

    dataset = SegmentationComparisonDataset(source_folder, image_transform=transform)
    dataset_loader = DataLoader(dataset, num_workers=8)

    model = model.to(device)

    model.eval()

    if dead_litter_as_bg:
        segmentation_class_names = class_names_nodl
    else:
        segmentation_class_names = class_names

    # len(class_names) + 1 due to uncertainty

    with torch.no_grad():
        miou = torchmetrics.IoU(len(segmentation_class_names) + 1)

        for i, (x, gt) in enumerate(dataset_loader):
            print("Image", i + 1)

            with torch.amp.autocast(device.type):
                if augmented_eval:
                    pred = model(
                        x.to(device),
                        mode="segmentation",
                        enriched_prediction=True,
                        interlaced_prediction=True,
                    )
                else:
                    pred = model(x.to(device), mode="segmentation")

            gt = gt[0]

            if dead_litter_as_bg:
                gt = merge_dead_litter_into_background(gt)

            if pred.shape != gt.shape:
                # Bilinear resizing to gt map
                print("Resizing", pred.shape, "to", gt.shape)
                pred = nn.functional.interpolate(pred, gt.shape[:2], mode="bilinear")
                #pred = tf.image.resize(pred, size=gt_map.shape[:2])

            pred = pred[0].to(torch.float32)

            pred = torch.moveaxis(pred.cpu(), 0, -1)
            pred = torch.argmax(pred, dim=-1)

            # Increase class index of every class by 1 to be on par with the GT and avoid the uncertainty class
            pred += 1

            miou.update(pred.to(torch.int32), gt.to(torch.int32))

        cm = miou.confmat

        # Remove uncertainty class
        cm = cm[1:, 1:]

        true_pos = torch.diag(cm)
        row_sums = torch.sum(cm, dim=0)
        col_sums = torch.sum(cm, dim=1)

        numerator = true_pos
        denominator = row_sums + col_sums - true_pos

        classwise_iou = numerator / denominator

        return segmentation_class_names.copy(), classwise_iou
