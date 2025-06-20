import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from .base_time_series_model import TorchTrainableTimeSeriesModel
from utils import torch_utils


class NStepPredictionModel(nn.Module):
    def __init__(self, spt_model, model_kwargs) -> None:
        super().__init__()

        self.spt_model = spt_model
        self.model_kwargs = model_kwargs

    def forward(self, x, pre_x_list):
        with torch.no_grad():
            pre_outs = [
                self.spt_model(pre_x, **self.model_kwargs)
                for pre_x in pre_x_list
            ]

            pre_outs = torch.stack(pre_outs, dim=0)

        out = self.spt_model(
            {"x": x, "suppl_data": pre_outs},
            **self.model_kwargs,
        )

        return out


class NStepPredictionTimeSeriesModel(TorchTrainableTimeSeriesModel):
    def __init__(self, spt_model, model_kwargs, dataset, **kwargs):
        super().__init__(**kwargs)

        self.model = NStepPredictionModel(
            spt_model,
            model_kwargs
        )
        self.dataset = dataset

    def _get_training_dataset(self, x, y, batch_size):
        indices = []

        for i, ln in enumerate(self.dataset.get_lengths()):
            indices.extend([(i, j) for j in range(ln)])

        return DataLoader(
            NPrevStepDatasetWrapper(self.dataset),
            sampler=SubsetRandomSampler(indices),
            batch_size=batch_size,
            shuffle=True,
        )

    def fit(self, x, y, seq_ids):
        torch_utils.set_trainability(self.model, True)
        torch_utils.set_trainability(self.model.feature_extractor, False)

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

