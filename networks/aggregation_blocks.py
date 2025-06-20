import torch

from torch import nn


def get_aggregation_block(feats, s):
    defs = s.split(",")

    args = {}

    for def_ in defs:
        if def_.startswith("s"):
            args["n_scales"] = int(def_[1:])
        elif def_.startswith("k"):
            args["kernel_size"] = int(def_[1:])
        elif def_.startswith("f"):
            args["dilation_factor"] = int(def_[1:])
        elif def_.startswith("n"):
            args["normalization"] = def_[1:]
        elif def_.startswith("u"):
            if def_[1:] == "c":
                args["fusion"] = "concat"
            elif def_[1:] == "s":
                args["fusion"] = "sum"
            elif def_[1:] == "sm":
                args["fusion"] = "summax"
            else:
                raise ValueError("Invalid fusion type: " + def_[1:])
        else:
            raise ValueError("Invalid specification: " + def_)

    return DilatedAggregationBlock(
        feats=feats,
        **args,
    )


# Similar to DeepLabV3 block
class DilatedAggregationBlock(nn.Module):
    def __init__(self, feats, n_scales, kernel_size, dilation_factor=2, normalization=None, fusion="concat"):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(
                feats,
                feats,
                kernel_size=(1, 1),
            )
        ])

        self.fusion = fusion

        for i in range(n_scales - 1):
            self.convs.append(
                nn.Conv2d(
                    feats,
                    feats,
                    kernel_size=kernel_size,
                    padding="same",
                    dilation=dilation_factor ** i,
                )
            )

        if normalization == "crelu":
            self.normalization = nn.Sequential(
                nn.Conv2d(feats, feats, kernel_size=(1, 1)),
                nn.ReLU(inplace=True),
            )
        elif normalization == "crelu2":
            self.normalization = nn.Sequential(
                nn.Conv2d(feats, feats, kernel_size=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(feats, feats, kernel_size=(1, 1)),
                nn.ReLU(inplace=True),
            )
        elif normalization == "csigm":
            self.normalization = nn.Sequential(
                nn.Conv2d(feats, feats, kernel_size=(1, 1)),
                nn.Sigmoid(),
            )
        elif normalization == "b":
            self.normalization = nn.Sequential(
                nn.BatchNorm2d(feats),
            )
        elif normalization == "g":
            self.normalization = nn.Sequential(
                nn.GroupNorm(16, feats),
            )
        elif normalization == "l":
            self.normalization = nn.Sequential(
                nn.GroupNorm(1, feats),
            )
        elif normalization == "i":
            self.normalization = nn.Sequential(
                nn.InstanceNorm2d(feats),
            )
        else:
            self.normalization = nn.Sequential()

        if fusion == "concat":
            self.final_conv = nn.Conv2d(
                n_scales * feats,
                feats,
                kernel_size=(1, 1),
            )
        else:
            self.final_conv = nn.Conv2d(
                feats,
                feats,
                kernel_size=(1, 1),
            )

    def forward(self, x):
        outputs = []

        for conv in self.convs:
            out = conv(x)
            out = torch.relu(out)

            outputs.append(out)

        if not hasattr(self, "fusion"):
            self.fusion = "concat"

        if self.fusion == "concat":
            final_out = torch.concat(outputs, dim=1)
        elif self.fusion == "sum":
            final_out = torch.sum(torch.stack(outputs, dim=0), dim=0)
        elif self.fusion == "summax":
            if self.training:
                final_out = torch.sum(torch.stack(outputs, dim=0), dim=0)
            else:
                final_out = torch.amax(torch.stack(outputs, dim=0), dim=0)
        else:
            raise ValueError()

        final_out = self.final_conv(final_out)
        final_out = torch.relu(final_out)

        if hasattr(self, "normalization"):
            final_out = self.normalization(final_out)

        return final_out