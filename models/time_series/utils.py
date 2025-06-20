import torch

from torch import nn


class FeatureFuser(nn.Module):
    def __init__(self, embedding_dimensions=128, normalization="layer", fuse_method="add"):
        super().__init__()

        self.embedding_dimensions = embedding_dimensions
        self.normalization = normalization
        self.fuse_method = fuse_method

        self.norm_layers = nn.ModuleDict()
        self.embedding_blocks = nn.ModuleDict()

    def __maybe_init_norm_layers(self, x):
        for idx, xinst in enumerate(x):
            if str(idx) not in self.norm_layers:
                if self.normalization in ("layer", "l"):
                    self.norm_layers[str(idx)] = nn.LayerNorm([xinst.shape[1], 1]) # xinst.shape[2]
                elif self.normalization in ("instance", "i"):
                    self.norm_layers[str(idx)] = nn.InstanceNorm1d(xinst.shape[1]) # xinst.shape[2]
                elif self.normalization in ("none", None):
                    pass
                else:
                    raise ValueError("Invalid norm layer: " + self.normalization)

    def __maybe_init_embeddings(self, x):
        if self.embedding_dimensions not in (0, "none", None):
            for idx, xinst in enumerate(x):
                if str(idx) not in self.embedding_blocks:
                    self.embedding_blocks[str(idx)] = nn.Sequential(
                        nn.Conv1d(xinst.shape[1], self.embedding_dimensions, kernel_size=1),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(self.embedding_dimensions, self.embedding_dimensions, kernel_size=1),
                        nn.ReLU(inplace=True),
                    )

    def __maybe_apply(self, x, block_dict):
        x_new = []

        for idx, xinst in enumerate(x):
            if str(idx) in block_dict:
                x_new.append(block_dict[str(idx)](xinst))
            else:
                x_new.append(xinst)

        return x_new

    def __maybe_apply_norm_layers(self, x):
        return self.__maybe_apply(x, self.norm_layers)

    def __maybe_apply_embeddings(self, x):
        return self.__maybe_apply(x, self.embedding_blocks)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]

        self.__maybe_init_norm_layers(x)
        x = self.__maybe_apply_norm_layers(x)

        self.__maybe_init_embeddings(x)
        x = self.__maybe_apply_embeddings(x)

        if len(x) == 1:
            return x[0]
        elif self.fuse_method == "concat":
            return torch.concat(x, dim=1)
        elif self.fuse_method == "add":
            return torch.sum(torch.stack(x, dim=0), dim=0)
        else:
            raise ValueError("Invalid fuse method: " + self.fuse_method)




