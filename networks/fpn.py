
import os
import torch
import torchvision
import torch.nn.functional as F

from torch import nn

from .aggregation_blocks import get_aggregation_block

# Get the number of output channels of a layer
def get_out_channels(layer):
    """
    This method currently works for Sequentials and EfficientNets. It might need to be extended/generalized for other architectures.
    """
    if isinstance(layer, nn.Sequential) or hasattr(layer, "layers") or hasattr(layer, "block"):
        layers = None
        if not isinstance(layer, nn.Sequential) and hasattr(layer, "block"):
            layer = layer.block

        if isinstance(layer, nn.Sequential):
            try:
                layers = layer[::-1]
            except TypeError:
                layers = [l for l in layer][::-1]
        elif hasattr(layer, "layers"):
            layers = layer.layers[::-1]
        else:
            raise ValueError("Invalid layer configuration")

        for sublayer in layers:
            if isinstance(sublayer, nn.Conv2d):
                return sublayer.out_channels
            elif isinstance(sublayer, nn.Linear):
                return sublayer.out_features
            elif isinstance(sublayer, nn.Sequential) or hasattr(sublayer, "layers") or hasattr(sublayer, "block"):
                c = get_out_channels(sublayer)

                if c is not None:
                    return c
        else:
            return None
    elif isinstance(layer, nn.Conv2d):
        return layer.out_channels
    elif isinstance(layer, nn.Linear):
        return layer.out_features
    else:
        raise ValueError("Unknown layer type: " + str(type(layer)))


class FPN(nn.Module):
    LATERAL_LAYER_NAMES = ["C0", "C1", "C2", "C3", "C4", "C5"]
    OUTPUT_LAYER_NAMES = ["P0", "P1", "P2", "P3", "P4", "P5"]

    def __init__(self, base_model, out_layer, d=256, use_bn=False, upsampling_method="nearest", aggregation="add", scaling_feats=False, output_aggregation=None, weights="IMAGENET1K_V1"):
        """
        Feature Pyramid Network (FPN) for semantic segmentation.

        Args:
            base_model (str): Name of the base model architecture (e.g., "resnet50") or path to a torch model file (.pth).
            out_layer (str or list): Name(s) of the output layer(s) to extract features from.
            d (int): Number of output channels for the FPN.
            use_bn (bool): Whether to use batch normalization.
            upsampling_method (str): Method for upsampling ("nearest", "bilinear", "transposed").
            aggregation (str): Method for aggregating features ("none", "add", "concat", "logsumexp", "gating").
            scaling_feats (bool): If true, the features will be scaled depending on their depth in the FPN. The deepest layers will have d features, halving for each depth layer in the FPN.
            output_aggregation (str or None): Method for aggregating output features ("none", "add", "mean").
            weights (str): Weights to use for the base model (e.g., "IMAGENET1K_V1").
        """
        assert out_layer in (self.LATERAL_LAYER_NAMES + self.OUTPUT_LAYER_NAMES) or isinstance(out_layer, (list, tuple)) and \
               all(out_layer_elem in (self.LATERAL_LAYER_NAMES + self.OUTPUT_LAYER_NAMES) for out_layer_elem in out_layer)
        # none aggregation just upscales the deepest layer features and hence
        # results in a simple decoder
        assert aggregation in ("none", "add", "concat", "logsumexp", "gating")
        assert output_aggregation in (None, "none", "add", "mean")
        super().__init__()

        self.use_bn = use_bn
        self.upsampling_method = upsampling_method
        self.aggregation_method = aggregation
        self.output_aggregation_method = output_aggregation

        self.p0_lateral_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(5, 5), padding="same"),
            nn.ReLU(inplace=True),
        )

        # Load base model from file if path is provided, otherwise try to load it from pytorch
        if os.path.isfile(base_model):
            print("Loaded model from", base_model)
            self.base_model = torch.load(base_model, weights_only=False)
        else:
            self.base_model = getattr(torchvision.models, base_model.lower())(weights=weights)

        self.out_layer = out_layer if isinstance(out_layer, (list, tuple)) else [out_layer]
        self.layers, self.layer_has_lateral_output = self._get_layers(
            base_model,
            self.base_model
        )

        # Adjust to models with fewer FPN layers
        n_fpn_layers = sum(self.layer_has_lateral_output)

        self.LATERAL_LAYER_NAMES = self.LATERAL_LAYER_NAMES[:n_fpn_layers]
        self.OUTPUT_LAYER_NAMES = self.OUTPUT_LAYER_NAMES[:n_fpn_layers]

        if not scaling_feats:
            self.d = [d] * len(self.OUTPUT_LAYER_NAMES)
        else:
            self.d = [d // (2 ** i) for i in range(len(self.OUTPUT_LAYER_NAMES))]#[::-1]

        self.lateral_convs, self.lateral_bns, self.output_convs, self.output_bns, self.up_convs, self.merge_convs = \
            self._build_lateral_connections(
                self.layers,
                self.layer_has_lateral_output
        )

        self.layers = nn.ModuleList(self.layers)
        self.lateral_convs = nn.ModuleList(self.lateral_convs)
        self.lateral_bns = nn.ModuleList(self.lateral_bns)
        self.output_convs = nn.ModuleList(self.output_convs)
        self.output_bns = nn.ModuleList(self.output_bns)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.merge_convs = nn.ModuleList(self.merge_convs)

        # Build a list containing all parameters of the FPN without base model
        # as central access point from the outside
        self.fpn_layers = nn.ModuleList()

        self.fpn_layers.append(self.p0_lateral_block)
        self.fpn_layers.extend(self.lateral_convs)
        self.fpn_layers.extend(self.lateral_bns)
        self.fpn_layers.extend(self.output_convs)
        self.fpn_layers.extend(self.output_bns)
        self.fpn_layers.extend(self.up_convs)
        self.fpn_layers.extend(self.merge_convs)

        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            self.initialized = True

            # Dry run for initialization of parameters
            with torch.no_grad():
                self(x)

            self._init_weights()

        output = dict()
        out_layers_to_extract = set(self.out_layer)
        largest_output = None

        lateral_outputs = []

        # Upward
        for i, (layer, is_lateral_output) in enumerate(zip(self.layers, self.layer_has_lateral_output)):
            x = layer(x)

            if is_lateral_output:
                if i == 0:
                    #lateral_outputs.append(self.p0_lateral_block(x))
                    # Lateral output for 0th layer is non-strided l1
                    if hasattr(self.layers[1][0], "stride"): # Only do this, if the layers as a strides attribute
                        s = self.layers[1][0].stride
                        self.layers[1][0].stride = (1, 1)
                        lateral_outputs.append(self.layers[1](x))
                        self.layers[1][0].stride = s
                    else: # Otherwise, ignore C0 lateral
                        lateral_outputs.append(None)
                else:
                    lateral_outputs.append(x)

        x = None

        # Downward
        for i, (lateral_name, output_name, lateral_output) in enumerate(
                zip(self.LATERAL_LAYER_NAMES[::-1], self.OUTPUT_LAYER_NAMES[::-1], lateral_outputs[::-1])):
            if lateral_name in out_layers_to_extract:
                output[lateral_name] = lateral_output
                out_layers_to_extract.discard(lateral_name)
                if not out_layers_to_extract:
                    break

            lateral_output = self.lateral_convs[::-1][i](lateral_output)
            if self.use_bn:
                lateral_output = self.lateral_bns[::-1][i](lateral_output)

            if x is None:
                if self.aggregation_method == "logsumexp":
                    x = torch.exp(lateral_output)
                else:
                    x = lateral_output
            else:
                # Upsample
                if self.upsampling_method == "transposed":
                    x = self.up_convs[::-1][i](x)
                else:
                    x = nn.Upsample(scale_factor=2, mode=self.upsampling_method)(x)

                # Merge with lateral
                if self.aggregation_method == "add":
                    x = x + lateral_output
                elif self.aggregation_method == "logsumexp":
                    x = x + torch.exp(lateral_output)
                elif self.aggregation_method == "concat":
                    x = torch.cat([x, lateral_output], dim=1)
                    x = self.merge_convs[::-1][i](x)
                    x = torch.relu(x)
                elif self.aggregation_method == "gating":
                    x = x * torch.sigmoid(lateral_output)

            if output_name in out_layers_to_extract:
                if self.aggregation_method == "logsumexp":
                    v = torch.log(x)
                else:
                    v = x

                v = self.output_convs[::-1][i](v)
                if self.use_bn:
                    v = self.output_bns[::-1][i](v)

                output[output_name] = torch.relu(v)
                if largest_output is None or \
                    v.shape[-1] * v.shape[-2] > \
                        largest_output[0] * largest_output[1]:
                    largest_output = v.shape[-2:]
                out_layers_to_extract.discard(output_name)
                if not out_layers_to_extract:
                    break

        if len(output) == 1:
            return list(output.values())[0]
        else:
            if self.output_aggregation_method not in (None, "none"):
                output = [
                    F.interpolate(
                        t[1],
                        size=largest_output,
                        mode="bilinear")
                    for t in sorted(output.items())
                ]

                if self.output_aggregation_method == "add":
                    output = torch.sum(torch.stack(output, dim=0), dim=0)
                elif self.output_aggregation_method == "mean":
                    output = torch.mean(torch.stack(output, dim=0), dim=0)

            return output

    def _get_layers(self, base_model_name, model):
        base_model_name = base_model_name.lower()

        layers = []

        if base_model_name in ("resnet50", ) or model.__class__.__name__.lower() in ("resnet",):
            l0 = nn.Sequential(
            )
            l1 = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
            )
            l1_5 = model.maxpool

            layers = [
                l0,
                l1,
                l1_5,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
            ]

            layer_has_lateral_output = [
                True,
                True,
                False,
                True,
                True,
                True,
                True,
            ]
        elif base_model_name in ("convnext_tiny", "convnext_small", "convnext_base", "convnext_large"):
            l0 = nn.Sequential()
            l1 = nn.Sequential(
                model.features[0],
                model.features[1],
            )
            l2 = nn.Sequential(
                model.features[2],
                model.features[3],
            )
            l3 = nn.Sequential(
                model.features[4],
                model.features[5],
            )
            l4 = nn.Sequential(
                model.features[6],
                model.features[7],
            )

            layers = [
                l0,
                l1,
                l2,
                l3,
                l4,
            ]

            layer_has_lateral_output = [
                True,
                True,
                True,
                True,
                True,
            ]
        elif base_model_name in ("efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l"):
            l0 = nn.Sequential(
            )
            l1 = nn.Sequential(
                model.features[0],
                model.features[1],
            )
            l2 = model.features[2]
            l3 = model.features[3]
            l4 = nn.Sequential(
                model.features[4],
                model.features[5],
            )
            max_layer_id = 7 if base_model_name == "efficientnet_v2_s" else 8
            l5 = nn.Sequential(
                *[
                    model.features[lidx] for lidx in range(6, max_layer_id + 1)
                ]
                # model.features[6],
                # model.features[7],
                # model.features[8],
            )

            layers = [
                l0,
                l1,
                l2,
                l3,
                l4,
                l5,
            ]

            layer_has_lateral_output = [
                True,
                True,
                True,
                True,
                True,
                True,
            ]
        else:
            raise ValueError("Unknown base model: " + base_model_name)

        return layers, layer_has_lateral_output

    def _build_lateral_connections(self, layers, layer_has_lateral_output):
        lateral_convs = []
        lateral_bns = []
        output_convs = []
        output_bns = []
        up_convs = []
        merge_convs = []

        layers = [layer for layer, has_lateral_output in zip(layers, layer_has_lateral_output) if has_lateral_output]

        for i, layer in enumerate(layers[::-1]):
            out_channels = get_out_channels(layer)

            if out_channels is None:
                out_channels = 0

            lateral_convs.append(nn.Conv2d(
                out_channels,
                self.d[i],
                kernel_size=(1, 1),
            ))
            output_convs.append(nn.Conv2d(
                self.d[i],
                self.d[i],
                kernel_size=(3, 3),
                padding="same",
            ))
            up_convs.append(nn.ConvTranspose2d(
                self.d[i],
                self.d[i],
                kernel_size=(2, 2),
                stride=(2, 2),
            ))
            merge_convs.append(nn.Conv2d(
                out_channels + self.d[i],
                self.d[i],
                kernel_size=(1, 1),
            ))

            if self.use_bn:
                lateral_bns.append(
                    nn.BatchNorm2d(self.d[i])
                )
                output_bns.append(
                    nn.BatchNorm2d(self.d[i])
                )

        return lateral_convs[::-1], lateral_bns[::-1], output_convs[::-1], output_bns[::-1], up_convs[::-1], merge_convs[::-1]

    def _build_lateral_connections_lazy(self, layers, layer_has_lateral_output):
        lateral_convs = []
        lateral_bns = []
        output_convs = []
        output_bns = []
        up_convs = []
        merge_convs = []

        layers = [layer for layer, has_lateral_output in zip(layers, layer_has_lateral_output) if has_lateral_output]

        for i, layer in enumerate(layers):
            lateral_convs.append(nn.LazyConv2d(
                #layer.out_channels,
                self.d[i],
                kernel_size=(1, 1),
            ))
            output_convs.append(nn.LazyConv2d(
                #self.d,
                self.d[i],
                kernel_size=(3, 3),
                padding="same",
            ))
            up_convs.append(nn.LazyConvTranspose2d(
                #self.d,
                self.d[i],
                kernel_size=(2, 2),
                stride=(2, 2),
            ))
            merge_convs.append(nn.LazyConv2d(
                #layer.out_channels,
                self.d[i],
                kernel_size=(1, 1),
            ))

            if self.use_bn:
                lateral_bns.append(
                    nn.BatchNorm2d(self.d[i])
                )
                output_bns.append(
                    nn.BatchNorm2d(self.d[i])
                )

        return lateral_convs[::-1], lateral_bns[::-1], output_convs[::-1], output_bns[::-1], up_convs[::-1], merge_convs[::-1]

    def _init_weights(self):
        for lateral_conv in self.lateral_convs:
            if not isinstance(lateral_conv.weight, nn.parameter.UninitializedParameter):
                nn.init.xavier_uniform_(lateral_conv.weight)
                nn.init.constant_(lateral_conv.bias, 0)

                print("Initialized weight for", lateral_conv)

        for output_conv in self.output_convs:
            if not isinstance(output_conv.weight, nn.parameter.UninitializedParameter):
                nn.init.xavier_uniform_(output_conv.weight)
                nn.init.constant_(output_conv.bias, 0)

                print("Initialized weight for", output_conv)


class AtrousFPN(FPN):
    def __init__(self, base_model, out_layer, d=256, use_bn=False, upsampling_method="nearest", aggregation="add", scaling_feats=False, atrous_spec=None):
        super().__init__(base_model, out_layer, d, use_bn, upsampling_method, aggregation, scaling_feats)

        if atrous_spec is not None:
            self.atrous_block = get_aggregation_block(d, atrous_spec)
        else:
            self.atrous_block = nn.Sequential()

        self.fpn_layers.append(self.atrous_block)

    def forward(self, x):
        x = super().forward(x)

        return self.atrous_block(x)

