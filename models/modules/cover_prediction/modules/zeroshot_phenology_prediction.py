import os
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from sklearn.cluster import DBSCAN
from skimage.segmentation import slic

from utils.augmentations.preprocessing import DePreprocess


def get_phenology_module(pheno_model):
    if os.path.exists(pheno_model):
        return PhenologyModelApplication(pheno_model)
    elif pheno_model in os.listdir("./zeroshot_phenology/models"):
        return PhenologyModelApplication(
            os.path.join("./zeroshot_phenology/models", pheno_model))
    else:
        raise ValueError("Invalid zeroshot pheno model: " + pheno_model)


class PhenologyImageSLICClustering(nn.Module):
    def __init__(self, preprocessing="caffe"):
        self.depreprocessor = DePreprocess(mode=preprocessing)

    def forward(self, imgs):

        for img in imgs:
            img = self.depreprocessor(img).cpu().numpy()
            img = np.moveaxis(img, 0, -1)

            # Run slic
            num_segments = 1500
            segments = slic(
                img,
                n_segments=num_segments,
                compactness=10,
                convert2lab=True,
                sigma=3
            )

            # Get means pixel values for each superpixel
            means = {}

            for i in set(segments.flatten()):
                means[i] = np.mean(img[segments == i], axis=0)

                #print(i, means[i])

            # Cluster superpixels by mean value
            keys = means.keys()
            labels = DBSCAN(eps=20).fit_predict(np.array(list(means.values())))

            pheno_map = np.ones_like(img)

            regular_keys = [k for k, label in zip(keys, labels) if label == 0]

            pheno_map[np.isin(segments, regular_keys)] = 0

            # img_new = img.copy()

            # for k, label in zip(keys, labels):
            #     print(k, label)
            #     if label == 0:
            #         img_new[segments == k] = 0

            return torch.tensor(pheno_map)




class PhenologyModelApplication(nn.Module):
    def __init__(self, model_path, scales=(1,)):
        super().__init__()

        self.model = torch.load(model_path, map_location="cpu", weights_only=False)
        self.scales = scales

    def forward(self, x):
        input_size = x.shape[-2:]

        maps = []

        for scale in self.scales:
            if scale != 1:
                x_new = F.interpolate(x, scale_factor=scale, mode="bilinear")
                x_out = self.model(x_new)
                x_out = F.interpolate(x_out, size=input_size, mode="bilinear")
            else:
                x_out = self.model(x)

            maps.append(x_out)

        if len(maps) == 1:
            return maps[0]
        else:
            return torch.mean(torch.stack(maps), 0)


