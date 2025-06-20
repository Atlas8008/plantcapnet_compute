import torch
import torch.nn.functional as F

from torch import nn



def get_data_injection_module(n_outputs, spec):
    """
    Create a DataInjectionModule based on the provided specifications.

    Parameters:
    - n_outputs (int): Number of output channels for the module.
    - spec (str): Comma-separated string specifying the module configuration. Each
      element in the string represents a configuration option, and the format is as follows:
        - 'bX': Set the number of blocks to X.
        - 'cX': Set the number of convolutions per block to X.
        - 'iX': Set the number of intermediate features to X.
        - 'ub': Use bilinear interpolation for upsampling.
        - 'ut': Use transposed convolution for upsampling.
        - 'fa': Use addition as the fusion method.
        - 'fc': Use concatenation as the fusion method.
        - 'prt': Apply ReLU before fusion.
        - 'prf': Do not apply ReLU before fusion.
        - 'nf': Do not apply fuse block after fusion.
        - 'ri': Resize inputs to feature size before applying CNN blocks.

    Returns:
    - DataInjectionModule: An instance of the DataInjectionModule class configured
      based on the provided specifications.

    Raises:
    - ValueError: If an invalid specification is encountered in the 'spec' argument.

    Example:
    ```python
    n_outputs = 128
    spec = "b3,c2,i64,ut,fc,prt"
    module_instance = get_data_injection_module(n_outputs, spec)
    ```
    """
    params = {}

    for s in spec.split(","):
        if s[0] == "b":
            params["n_blocks"] = int(s[1:])
        elif s[0] == "c":
            params["n_convs_per_block"] = int(s[1:])
        elif s[0] == "i":
            params["n_inter_feats"] = int(s[1:])
        elif s[0] == "t":
            params["td_embedding"] = s[1:]
        elif s == "ub":
            params["upsampling"] = "bilinear"
        elif s == "ut":
            params["upsampling"] = "transposed"
        elif s == "fa":
            params["fuse_method"] = "add"
        elif s == "fc":
            params["fuse_method"] = "concat"
        elif s == "prt":
            params["pre_fuse_relu"] = True
        elif s == "prf":
            params["pre_fuse_relu"] = False
        elif s == "nf":
            params["apply_fuse_block"] = False
        elif s == "ri":
            params["resize_inputs"] = True
        else:
            raise ValueError("Invalid spec argument: " + s)

    return DataInjectionModule(
        n_outputs=n_outputs,
        **params
    )


class LazyTSEmbeddingModel(nn.Module):
    def __init__(self, kind, out_feats, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.kind = kind
        self.out_feats = out_feats

        self.initialized = False

    def _maybe_initialize(self, x):
        if self.kind == "lstm":
            self.model = nn.LSTM(
                input_size=x.shape[-1],
                hidden_size=self.out_feats,
                batch_first=True,
            ).to(x.device)

        self.initialized = True

    def forward(self, x):
        x = torch.swapaxes(x, 1, 2)

        self._maybe_initialize(x)

        h0 = torch.zeros((1, x.shape[0], self.out_feats)).to(x.device)

        if self.kind == "lstm":
            c0 = torch.zeros((1, x.shape[0], self.out_feats)).to(x.device)
            x, _ = self.model(x, (h0, c0))
        else:
            raise ValueError("Invalid TSEmbedding kind:", self.kind)

        # Only retain the output of the last element of the sequence
        x = x[:, -1]

        return x





class TimeDelayEmbeddingModule(nn.Module):
    def __init__(self, kind, out_feats, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if kind == "fc":
            self.embedder = nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(
                    out_features=out_feats
                ),
            )
        elif kind == "lstm":
            self.embedder = LazyTSEmbeddingModel(
                kind=kind,
                out_feats=out_feats
            )

    def forward(self, x):
        # Input shape should be (B, N, T)
        x = self.embedder(x)

        return x


class DataInjectionModule(nn.Module):
    def __init__(self, n_outputs, n_blocks, n_convs_per_block, n_inter_feats=None, upsampling="transposed", fuse_method="add", pre_fuse_relu=True, apply_fuse_block=True, td_embedding=None, resize_inputs=False):
        super().__init__()

        self.fuse_method = fuse_method
        self.n_outputs = n_outputs
        self.pre_fuse_relu = pre_fuse_relu
        self.apply_fuse_block = apply_fuse_block
        self.resize_inputs = resize_inputs

        if td_embedding is not None:
            self.time_delay_embedding_module = TimeDelayEmbeddingModule(
                td_embedding,
                out_feats=n_inter_feats,
            )
        else:
            self.time_delay_embedding_module = None

        self.data_decoder = self.build_decoder(
            n_outputs=n_outputs,
            n_inter_feats=n_inter_feats or n_outputs,
            n_blocks=n_blocks,
            n_convs_per_block=n_convs_per_block,
            upsampling=upsampling,
        )

        self.fuse_block = nn.Sequential(
            nn.Conv2d(
                n_outputs if fuse_method == "add" else 2 * n_outputs,
                n_outputs,
                (1, 1)
            ),
            nn.ReLU(),
            nn.Conv2d(n_outputs, n_outputs, (1, 1)),
            nn.ReLU(),
        )


    def build_decoder(self, n_outputs, n_inter_feats, n_blocks, n_convs_per_block, upsampling):

        layers = []

        for i in range(n_blocks + 1):
            for j in range(n_convs_per_block):
                kernel_size = 1 if i == 0 else 3
                input_channels = n_inter_feats
                output_channels = n_inter_feats if i < n_blocks or j < n_convs_per_block - 1 else n_outputs

                if (i == 0 and j == 0):
                    layers.append(nn.LazyConv2d(output_channels, kernel_size=kernel_size, padding="same"))
                else:
                    layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding="same"))
                if self.pre_fuse_relu or \
                    not (i == n_blocks and j == n_convs_per_block - 1):
                    layers.append(nn.ReLU(inplace=True))

            if i < n_blocks:
                if upsampling == "bilinear":
                    layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
                elif upsampling == "transposed":
                    layers.append(nn.ConvTranspose2d(n_inter_feats, n_inter_feats, kernel_size=(2, 2), stride=(2, 2)))
                else:
                    raise ValueError("Invalid upsampling method: " + upsampling)

        return nn.Sequential(*layers)

    def forward(self, x, feats):
        if self.time_delay_embedding_module is not None:
            x = self.time_delay_embedding_module(x)

        if hasattr(self, "resize_inputs") and self.resize_inputs:
            x = F.interpolate(
                x[..., None, None],
                size=feats.shape[2:],
                mode="bilinear",
            )
        else:
            x = x[..., None, None]

        #data_feats = self.data_decoder(x[..., None, None])
        data_feats = self.data_decoder(x)

        if not all(a == b for a, b in zip(data_feats.shape[2:], feats.shape[2:])):
            data_feats = F.interpolate(
                data_feats,
                size=feats.shape[2:],
                mode="bilinear",
            )

        if self.fuse_method == "add":
            x = feats + data_feats
        elif self.fuse_method == "concat":
            x = torch.concat([feats, data_feats], dim=1)
        else:
            raise ValueError("Invalid fuse method: " + self.fuse_method)

        if hasattr(self, "apply_fuse_block") and self.apply_fuse_block:
            return self.fuse_block(x)
        else:
            return x



