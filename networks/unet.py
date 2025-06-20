import torch
from torch import nn


class UNet(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            features,
            use_skips=True,
            use_bn=False,
            convs_per_step=2,
            downsampling_method="conv",
            upsampling_method="bilinear",
            activation_function="relu",
            final_activation=None,
        ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.use_skips = use_skips
        self.use_bn = use_bn
        self.convs_per_step = convs_per_step
        self.downsampling_method = downsampling_method
        self.upsampling_method = upsampling_method
        self.final_activation = final_activation

        if activation_function == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation_function == "leaky_relu":
            self.act = nn.LeakyReLU()
        else:
            raise ValueError("Invalid activation function: " + activation_function)

        self.down_blocks = nn.ModuleList(self._build_down())
        self.up_blocks = nn.ModuleList(self._build_up())

    def forward(self, x):
        skips = []

        for i, block in enumerate(self.down_blocks):
            x = block(x)

            if i < len(self.down_blocks) - 1:
                skips.append(x)

        for i, (skip, block) in enumerate(zip(skips[::-1], self.up_blocks)):
            if self.use_skips:
                x = torch.cat([x, skip], dim=1)

            x = block(x)

        return x

    def _build_down(self):
        blocks = []

        initial_block = self._conv_block(self.in_channels, self.features[0])

        blocks.append(nn.Sequential(*initial_block))

        for i, out_feats in enumerate(self.features[1:], start=1):
            blocks.append(self._build_single_down(
                self.features[i - 1], out_feats, final_down_block=i == len(self.features) - 1))

        return blocks

    def _build_up(self):
        blocks = []

        for i, out_feats in enumerate(self.features[::-1][1:], start=1):
            out_feats_ = out_feats# if i < len(self.features) - 1 else self.out_channels
            in_feats_ = self.features[::-1][i - 1] if not self.use_skips else self.features[::-1][i - 1] + self.features[::-1][i]

            blocks.append(self._build_single_up(in_feats_, out_feats_, final_up_block=i == len(self.features) - 1))

        return blocks

    def _build_single_down(self, in_feats, out_feats, final_down_block=False):
        stride_downsampling = self.downsampling_method == "conv"

        layers = []

        if self.downsampling_method == "max":
            layers.append(nn.MaxPool2d((2, 2)))
        elif self.downsampling_method == "avg":
            layers.append(nn.AvgPool2d((2, 2)))
        else:
            raise NotImplementedError("Not implemented downsampling method: " + self.downsampling_method)

        layers.extend(self._conv_block(in_feats, out_feats, stride_downsampling=stride_downsampling))

        if final_down_block:
            if self.upsampling_method in ("nearest", "bilinear"):
                layers.append(nn.Upsample(scale_factor=(2, 2), mode=self.upsampling_method))
            elif self.upsampling_method == "transposed":
                layers.append(nn.ConvTranspose2d(out_feats, out_feats, kernel_size=(2, 2), stride=2))
            else:
                raise NotImplementedError("Not implemented upsampling method: " + self.upsampling_method)

        return nn.Sequential(*layers)

    def _build_single_up(self, in_feats, out_feats, final_up_block=False):
        layers = []

        layers.extend(self._conv_block(in_feats, out_feats, final_out_feats=self.out_channels if final_up_block else None))

        if not final_up_block:
            if self.upsampling_method in ("nearest", "bilinear"):
                layers.append(nn.Upsample(scale_factor=(2, 2), mode=self.upsampling_method))
            elif self.upsampling_method == "transposed":
                layers.append(nn.ConvTranspose2d(out_feats, out_feats, kernel_size=(2, 2), stride=2))
            else:
                raise NotImplementedError("Not implemented upsampling method: " + self.upsampling_method)

        return nn.Sequential(*layers)

    def _conv_block(self, in_feats, out_feats, final_out_feats=None, stride_downsampling=False):
        layers = []

        for i in range(self.convs_per_step):
            in_feats_ = in_feats if i == 0 else out_feats
            out_feats_ = out_feats if i < self.convs_per_step - 1 or final_out_feats is None else final_out_feats
            stride = 2 if stride_downsampling and i == 0 else 1

            if stride == 2:
                layers.append(nn.ZeroPad2d((1, 0, 1, 0)))
            else:
                layers.append(nn.ZeroPad2d((1, 1, 1, 1)))

            layers.append(nn.Conv2d(in_feats_, out_feats_, kernel_size=(3, 3), stride=stride, padding="valid"))
            if self.use_bn:
                layers.append(nn.BatchNorm2d(out_feats))
            if final_out_feats and self.final_activation and i == self.convs_per_step - 1:
                layers.append(self.final_activation)
            else:
                layers.append(self.act)

        return layers

