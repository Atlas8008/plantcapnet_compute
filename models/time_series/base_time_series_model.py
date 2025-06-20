import torch
import random
import numpy as np
import torch.nn.functional as F

from scipy.interpolate import interp1d
from math import floor, ceil

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanSquaredError

from abc import ABC, abstractmethod


def interpolate_seq_based(x, seq_ids, min_id=None, max_id=None):
    if min_id is None:
        min_id = min(seq_ids)
    if max_id is None:
        max_id = max(seq_ids) + 1

    seq_new = np.arange(min_id, max_id)

    x_new = interp1d(seq_ids, np.array(x).T, fill_value="extrapolate")(seq_new).T.astype("float32")

    return x_new




class TSTrainingDataset(Dataset):
    def __init__(self, x, y, seq_ids=None, seq_based_interpolation=False, rdm_slice_size=None, pad_to=None) -> None:
        super().__init__()

        self.x = x
        self.y = y
        self.seq_ids = seq_ids
        self.seq_based_interpolation = seq_based_interpolation
        self.rdm_slice_size = rdm_slice_size
        self.pad_to = pad_to

    def __getitem__(self, index):
        x_full, y_full = self.x[index], self.y[index]

        seq_ids = None

        if isinstance(x_full, tuple):
            x_full = list(x_full)
        else:
            x_full = [x_full]

        if isinstance(y_full, tuple):
            y_full = list(y_full)
        else:
            y_full = [y_full]

        if self.seq_based_interpolation:
            seq_ids = self.seq_ids[index]

            min_id = min(sid for sid_lst in self.seq_ids for sid in sid_lst)
            max_id = max(sid for sid_lst in self.seq_ids for sid in sid_lst)

            seq_new = np.arange(min_id, max_id + 1)

            x_full = [interpolate_seq_based(xv, seq_ids, min_id=min_id, max_id=max_id + 1) for xv in x_full]
            y_full = [interpolate_seq_based(yv, seq_ids, min_id=min_id, max_id=max_id + 1) for yv in y_full]

            seq_ids = seq_new
        elif self.seq_ids is not None:
            seq_ids = self.seq_ids[index]

        x_full = [torch.tensor(x) for x in x_full]
        y_full = [torch.tensor(y) for y in y_full]

        if self.pad_to is not None and x_full[0].shape[0] < self.pad_to:
            diff = (self.pad_to - x_full[0].shape[0]) / 2

            diff_l = int(ceil(diff))
            diff_r = int(floor(diff))

            x_full = [F.pad(x[None], (0, 0, diff_l, diff_r), mode="replicate")[0] for x in x_full]
            y_full = [F.pad(y[None], (0, 0, diff_l, diff_r), mode="replicate")[0] for y in y_full]

        if self.rdm_slice_size is not None:
            assert x_full[0].shape[0] >= self.rdm_slice_size, f"The input shape is smaller than the requested slice size. ({x_full.shape[0]} < {self.rdm_slice_size})"

            slice_start = random.randint(0, x_full.shape[0] - self.rdm_slice_size)

            x_full = [x[slice_start:slice_start + self.rdm_slice_size] for x in x_full]
            y_full = [y[slice_start:slice_start + self.rdm_slice_size] for y in y_full]

        if len(x_full) == 1:
            x_full = x_full[0]
        if len(y_full) == 1:
            y_full = y_full[0]

        if seq_ids is not None:
            return x_full, y_full, seq_ids

        return x_full, y_full

    def __len__(self):
        return len(self.x)


class BaseTimeSeriesModel(ABC):
    def __init__(self):
        super().__init__()

    @staticmethod
    def clip(x, clip):
        if isinstance(clip, tuple):
            x = torch.clip(x, *clip)
        else:
            raise ValueError("Invalid clip value:", clip)

        return x

    @abstractmethod
    def __call__(self, x_list, clip):
        pass


class TrainableTimeSeriesModel(BaseTimeSeriesModel):
    def __init__(self):
        self.requires_training = True
        super().__init__()

    @abstractmethod
    def fit(self, x, y, seq_ids):
        pass


class TorchTrainableTimeSeriesModel(TrainableTimeSeriesModel):
    def __init__(self, train_epochs=50, train_batch_size=1, train_lr=0.1, train_loss="mae", weight_decay=0.0, train_pad=None, train_slice=None):
        super().__init__()

        self.model = None
        self.train_epochs = train_epochs
        self.train_batch_size = train_batch_size
        self.train_lr = train_lr
        self.train_loss = train_loss
        self.weight_decay = weight_decay
        self.train_pad = train_pad
        self.train_slice = train_slice

    def __str__(self) -> str:
        return self.__class__.__name__ + ":\n" + str(self.model)

    def _get_training_dataset(self, x, y, batch_size):
        return DataLoader(
            TSTrainingDataset(
                x,
                y,
                pad_to=self.train_pad,
                rdm_slice_size=self.train_slice
            ),
            batch_size=batch_size,
            shuffle=True,
        )

    def _get_loss(self, loss):
        if loss == "mae":
            return nn.L1Loss()
        elif loss == "mse":
            return nn.MSELoss()
        elif loss == "rmse":
            return MeanSquaredError(squared=False)
        else:
            raise ValueError("Invalid loss function: " + loss)

    def fit(self, x, y, seq_ids):
        self.model = self._train(
            model=self.model,
            x=x,
            y=y,
            epochs=self.train_epochs,
            loss=self.train_loss,
            lr=self.train_lr,
            batch_size=self.train_batch_size,
            weight_decay=self.weight_decay,
        )

    def _train(self, model, x, y, epochs, loss, lr=0.1, batch_size=1, weight_decay=0.0):
        loader = self._get_training_dataset(x, y, batch_size=batch_size)

        # Init lazy weights
        with torch.no_grad():
            model.eval()
            model(next(iter(loader))[0])

        model.train()
        loss_fn = self._get_loss(loss)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            gamma=0.1,
            milestones=[int(epochs * 0.8), int(epochs * 0.95)],
        )

        for epoch in range(epochs):
            loss_mean = 0
            count = 0

            for _x, _y in loader:
                model.zero_grad()

                predictions = model(_x)

                loss = loss_fn(predictions, _y)
                loss.backward()

                optimizer.step()

                with torch.no_grad():
                    loss_mean += loss
                    count += 1

            scheduler.step()

            if (epoch + 1) in [int(i * 0.1 * epochs) for i in range(11)]:
                with torch.no_grad():
                    print(f"Epoch {epoch + 1}/{epochs}, loss:", loss_mean / count)

        return model

    @torch.no_grad()
    def __call__(self, x_list, seq_ids=None, clip=None):
        self.model.eval()
        if isinstance(x_list, tuple):
            x_list = tuple(torch.tensor(xl)[None] for xl in x_list)
        else:
            x_list = torch.tensor(x_list)[None]
        result = self.model(x_list)[0]
        result = self.clip(result, clip)

        return result.numpy()

