import sys
import torch
import warnings
import numpy as np
import pandas as pd
import torchmetrics

from statsmodels.formula.api import ols

np.set_printoptions(threshold=sys.maxsize)


class AncovaR2(torchmetrics.Metric):
    """ANCOVA R2 metric.
    This metric computes the R2 value of a linear regression model
    with the given x and y values. The model is fitted using
    ordinary least squares (OLS) regression, and the R2 value is
    computed using the statsmodels library.
    """
    higher_is_better = True
    _expensive = True

    def __init__(self, mean_over_last_dim=False, class_mask=None, last_dim_idx=None, name="", order="op") -> None:
        """
        Args:
            mean_over_last_dim (bool): If True, the metric will be computed
                over the last dimension of the input tensors.
            class_mask (list): A list of boolean values indicating which
                classes to include in the metric computation.
            last_dim_idx (int): The index of the last dimension to compute
                the metric over.
            name (str): The name of the metric.
            order (str): The order of the x and y values. Can be "op" or "po" (Observed/Predicted or Predicted/Observed).

        """
        super().__init__(dist_sync_on_step=False)

        self.mean_over_last_dim = mean_over_last_dim
        self.last_dim_idx = last_dim_idx
        self.name = name
        self.order = order

        if class_mask is not None:
            self.class_mask = torch.tensor(class_mask, dtype=torch.bool)
        else:
            self.class_mask = class_mask

        self.add_state("x", default=list(), dist_reduce_fx="cat")
        self.add_state("y", default=list(), dist_reduce_fx="cat")

    @torch.jit.unused
    def forward(self, *args, **kwargs):
        self.update(*args, **kwargs)

        self._forward_cache = torch.nan

        return self._forward_cache

    def update(self, x, y):
        if self.class_mask is not None:
            self.x.append(x[:, self.class_mask])
            self.y.append(y[:, self.class_mask])
        else:
            self.x.append(x)
            self.y.append(y)

    @torch.no_grad()
    def compute(self):
        x_full = torch.vstack(self.x).cpu().numpy()
        y_full = torch.vstack(self.y).cpu().numpy()

        assert self.order in ("op", "po")

        if self.order == "op":
            x_full, y_full = y_full, x_full

        if self.mean_over_last_dim is None:
            x_vals = [x_full]
            y_vals = [y_full]
        else:
            if self.last_dim_idx is not None:
                x_vals = [x_full[..., self.last_dim_idx]]
                y_vals = [y_full[..., self.last_dim_idx]]
            else:
                x_vals = [x_full[..., i] for i in range(x_full.shape[-1])]
                y_vals = [y_full[..., i] for i in range(y_full.shape[-1])]


        results = []

        for x, y in zip(x_vals, y_vals):
            nlabels = np.arange(x.shape[-1])
            nlabels = np.tile(nlabels, x.shape[0])

            x = x.flatten()
            y = y.flatten()

            df = pd.DataFrame({
                    "x": x,
                    "y": y,
                    "labels": nlabels,
                }
            )

            formula = "y ~ x * C(labels)" #

            with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
                model = ols(formula, data=df).fit()

                results.append(model.rsquared)

        return np.mean(results)