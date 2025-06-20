import torch
import numpy as np
import pandas as pd
from datetime import datetime

try:
    from torchcast.kalman_filter import KalmanFilter
    from torchcast.utils import TimeSeriesDataset, TimeSeriesDataLoader, complete_times
    from torchcast.process import LocalLevel, LocalTrend, Season
    from torchcast.covariance import Covariance
except ImportError as e:
    import warnings

    warnings.warn("Torchcast import failed. If used, torchcast should be installed correctly beforehand.")

from torch.utils.data import Dataset, DataLoader

from collections import defaultdict

from .base_time_series_model import TrainableTimeSeriesModel, TSTrainingDataset, interpolate_seq_based


class KalmanFilterTimeSeriesModel(TrainableTimeSeriesModel):
    def __init__(self, trend_kinds, n_outputs, fit_each_prediction=False, process_noise=None, measurement_noise=None, velocity_multi=0.1, season_period=None, season_K=None, no_fit=False, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(trend_kinds, (tuple, list)):
            trend_kinds = [trend_kinds]


        measure_dummy_names = [str(i) for i in range(n_outputs)]

        self.trend_kinds = trend_kinds
        self.measures = measure_dummy_names
        self.fit_each_prediction = fit_each_prediction
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.velocity_multi = velocity_multi
        self.season_period = season_period
        self.season_K = season_K
        self.no_fit = no_fit
        self.sequence_based_interpolation = False
        self.df_datasets = True

        self.requires_training = not no_fit

        if not fit_each_prediction:
            self.model = self._get_kalman_filter()
        else:
            self.model = None

    def __str__(self) -> str:
        return super().__str__() + str({k: v for k, v in self.__dict__.items() if not k.startswith("_")})

    def _get_kalman_filter(self):
        processes = []

        for measure in self.measures:
            for trend_kind in self.trend_kinds:
                if trend_kind == "ll":
                    trend = LocalLevel(measure + "_level", measure=measure)
                elif trend_kind == "lt":
                    trend = LocalTrend(
                        measure + "_trend",
                        measure=measure,
                        velocity_multi=self.velocity_multi,
                    )
                elif trend_kind == "s":
                    trend = Season(
                        measure + "_season",
                        dt_unit=None,
                        measure=measure,
                        K=self.season_K,
                        period=self.season_period,
                        fixed=True,
                    )
                else:
                    raise ValueError("Invalid trend kind: " + trend_kind)

                processes.append(
                    trend
                )

        if self.process_noise is not None:
            covp = Covariance.from_processes(
                processes,
                cov_type="process",
                init_diag_multi=self.process_noise,
            )
        else:
            covp = None

        if self.measurement_noise is not None:
            covm = Covariance.from_measures(
                self.measures,
                init_diag_multi=self.measurement_noise,
            )
        else:
            covm = None


        return KalmanFilter(
            measures=self.measures,
            processes=processes,
            process_covariance=covp,
            measure_covariance=covm,
        )

    def _ts_list_as_dataset(self, x, seq_ids):
        group_rows = []
        x_rows = defaultdict(list)
        seq_rows = []
        is_orig_rows = []

        for i, (xi, seqi) in enumerate(zip(x, seq_ids)):
            if isinstance(xi, tuple):
                assert len(xi) == 1
                xi = xi[0]

            assert len(xi) == len(seqi)
            xi = np.array(xi)

            for measure in self.measures:
                x_rows[measure].extend(xi[:, int(measure)])
            seq_rows.extend(np.array(seqi).astype(int))
            group_rows.extend([i] * len(xi))
            is_orig_rows.extend([True] * len(xi))

        df = pd.DataFrame({
            "groups": group_rows,
            "time_idx": seq_rows,
            "is_orig": is_orig_rows,
            **x_rows,
        })


        df["times"] = pd.to_timedelta(df["time_idx"], unit="W") + datetime(2020, 1, 1)

        df = complete_times(
            df,
            group_colnames="groups",
            time_colname="times",
            dt_unit="W",
        )
        df["is_orig"] = df["is_orig"].fillna(False)

        is_orig_data_point = list(df["is_orig"]) #[t in seq_ids for t in df["time_idx"]]

        print(df)

        return TimeSeriesDataset.from_dataframe(
            df,
            group_colname="groups",
            time_colname="times",
            dt_unit="W",
            measure_colnames=self.measures,
        ), is_orig_data_point

    def _get_training_dataset(self, x, y, seq_ids):
        if self.df_datasets:
            return self._ts_list_as_dataset(x, seq_ids)[0]
        else:
            # If the x values are tuples, check that they only contain a single value
            assert all(not isinstance(xi, tuple) or len(xi) == 1 for xi in x), "Kalman filters are only applicable to single output values, not combinations"

            seq_ids_old = seq_ids

            if not self.sequence_based_interpolation:
                seq_ids_pre = None
            else:
                seq_ids_pre = seq_ids

            data = next(iter(DataLoader(
                TSTrainingDataset(
                    x,
                    y,
                    seq_ids=seq_ids_pre,
                    seq_based_interpolation=self.sequence_based_interpolation,
                    pad_to=max(len(xi[0]) for xi in x),
                ),
                batch_size=len(x),
            )))

            if self.sequence_based_interpolation:
                x_full_batched, _, seq_ids = data
            else:
                x_full_batched, _ = data

            # Check correctness, if there are different (interpolated) sequence ids
            # We check, that for the same sequence ids, the interpolated values are the same, as they should not be changed during interpolation
            if not all(sid_old == sid for sid_old, sid in zip(seq_ids_old, seq_ids)):
                old_values = []
                new_values = []

                for x_, seq_ids_ in zip(x, seq_ids_old):
                    old_values.append(
                        {sid: xv for sid, xv in zip(seq_ids_, x_[0])}
                    )

                for x_, seq_ids_ in zip(x_full_batched, seq_ids):
                    new_values.append(
                        {int(sid): xv.numpy() for sid, xv in zip(seq_ids_, x_)}
                    )

                # Check, if the original values were kept in the interpolated dataset
                for ts_id, (ts_old, ts_new) in enumerate(zip(old_values, new_values)):
                    for sid in ts_old.keys():
                        old, new = ts_old[sid], ts_new[sid]

                        assert np.allclose(old, new), f"Values {old} != {new}, (TSID={ts_id}, SID={sid}, len={len(ts_old)})\nEntire series: \nOld:\n{ts_old}\n\nNew:\n{ts_new}"

            # We interpolate and extrapolate according to sequence_ids, so every time series starts with seq_id 0
            return TimeSeriesDataset(
                x_full_batched,
                group_names=[f"site_{i}" for i in range(len(x))],
                start_times=[[0] for series in x],
                measures=[self.measures],
                dt_unit=None,
            )

    def fit(self, x, y, seq_ids):
        if not self.fit_each_prediction:
            self._fit_model(
                x,
                y,
                seq_ids,
                self.model,
            )

    def _fit_model(self, x, y, seq_ids, model):
        if not self.no_fit:
            model.fit(
                self._get_training_dataset(
                    x,
                    y,
                    seq_ids,
                ).tensors[0], # Zero index, because the function returns a len 1 tuple
            )


    @torch.no_grad()
    def __call__(self, x_list, seq_ids, clip=None):
        if not self.fit_each_prediction:
            model = self.model
        else:
            model = self._get_kalman_filter()

            self._fit_model(
                [x_list],
                [x_list],
                [seq_ids],
                model,
            )

        is_orig_data_point = None

        if self.sequence_based_interpolation:
            seq_ids_new = np.arange(min(seq_ids), max(seq_ids) + 1)
            x_tensors = [torch.tensor(interpolate_seq_based(x, seq_ids))[None] for x in x_list]
        else:
            seq_ids_new = seq_ids
            if self.df_datasets:
                ds, is_orig_data_point = self._ts_list_as_dataset([x_list], [seq_ids])

                print(is_orig_data_point)


                x_tensors = [ds.tensors[0]]
            else:
                x_tensors = [torch.tensor(x)[None] for x in x_list]

        result = [
            model(
                x,
                start_offsets=[min(seq_ids)],
        ).means for x in x_tensors]

        # As we entered interpolated results as inputs to cover for missing values, we now have to select the correct desired values based on the original seq_ids
        if is_orig_data_point is None:
            is_orig_data_point = [sid in seq_ids for sid in seq_ids_new]
        print("Len before selecting:", result[0].shape[1])
        result = [r[:, is_orig_data_point] for r in result]
        print("Len after selecting:", result[0].shape[1])
        result = torch.concat(result, dim=0)[0]
        #assert len(result[0]) == len(x_list[0])
        result = self.clip(result, clip)

        return result.numpy()