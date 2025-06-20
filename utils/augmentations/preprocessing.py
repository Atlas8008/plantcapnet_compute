from copy import deepcopy
import torch

from torch import nn
from torchvision import transforms


class Preprocess(nn.Module):
    def __init__(self, mode="torch", preceded_by_scaling=True):
        assert mode in ("torch", "tf", "caffe"), "Unknown mode: " + str(mode)
        super().__init__()

        self.preceded_by_scaling = preceded_by_scaling
        self.mode = mode
        if self.mode == "torch":
            self.normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif self.mode == "tf":
            self.normalization = None
        else: # Caffe
            self.normalization = transforms.Normalize(mean=[103.939, 116.779, 123.68], std=[1.0, 1.0, 1.0])

    def forward(self, tensor):
        if self.mode == "caffe":
            # RGB --> BGR
            if len(tensor.shape) == 3:
                tensor = torch.flip(tensor, dims=(0, ))
            elif len(tensor.shape) == 4:
                tensor = torch.flip(tensor, dims=(1, ))
            else:
                raise ValueError(f"Invalid tensor rank: {tensor.shape}, tensor should have either rank 3 or 4")

            # Unscale values from ToTensor
            if self.preceded_by_scaling:
                tensor = tensor * 255.0
        else:
            if not self.preceded_by_scaling:
                tensor = tensor / 255.0

        if self.mode == "tf":
            # Zero-center values
            tensor = tensor * 2
            #tensor *= 2
            tensor -= 1

        if self.normalization is not None:
            tensor = self.normalization(tensor)

        return tensor



class DePreprocess(nn.Module):
    def __init__(self, mode="torch"):
        assert mode in ("torch", "tf", "caffe"), "Unknown mode: " + str(mode)
        super().__init__()

        self.mode = mode

        if self.mode == "torch":
            self.means = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        elif self.mode == "tf":
            self.means = None
            self.std = None
        else: # Caffe
            self.means = torch.tensor([103.939, 116.779, 123.68])[:, None, None]
            self.std = None

    def forward(self, tensor):
        tensor = deepcopy(tensor)

        if self.std is not None:
            tensor *= self.std.to(tensor.device)
        if self.means is not None:
            tensor += self.means.to(tensor.device)

        if self.mode == "caffe":
            # BGR --> RGB
            if len(tensor.shape) == 3:
                tensor = torch.flip(tensor, dims=(0, ))
            elif len(tensor.shape) == 4:
                tensor = torch.flip(tensor, dims=(1, ))
            else:
                raise ValueError(f"Invalid tensor rank: {tensor.shape}, tensor should have either rank 3 or 4")
        elif self.mode == "tf":
            # Zero-center values
            tensor += 1
            tensor /= 2

        if self.mode in ("tf", "torch"):
            tensor *= 255

        return tensor

if __name__ == "__main__":
    im = torch.randint(0, 255, (3, 768, 1024)).to(torch.float32)
    imgs = torch.randint(0, 255, (1, 3, 768, 1024)).to(torch.float32)

    caffe_pp = Preprocess("caffe", preceded_by_scaling=False)
    caffe_dp = DePreprocess("caffe")

    torch_pp = Preprocess("torch", preceded_by_scaling=False)
    torch_dp = DePreprocess("torch")

    tf_pp = Preprocess("tf", preceded_by_scaling=False)
    tf_dp = DePreprocess("tf")

    def compare(a, b):
        a = torch.mean(a)
        b = torch.mean(b)

        assert torch.isclose(a, b), f"{a.item()} != {b.item()}"

    compare(im, caffe_dp(caffe_pp(im)))
    compare(im, torch_dp(torch_pp(im)))
    compare(im, torch_dp(torch_pp(caffe_dp(caffe_pp(im)))))

    compare(imgs, caffe_dp(caffe_pp(imgs)))
    compare(imgs, torch_dp(torch_pp(imgs)))
    compare(imgs, torch_dp(torch_pp(caffe_dp(caffe_pp(imgs)))))