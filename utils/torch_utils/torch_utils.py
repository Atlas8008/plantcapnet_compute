import os
import sys
import time
import numpy as np
import torch
import shutil
import hashlib

from utils import jupyter_utils

IS_JUPYTER = not jupyter_utils.check_if_script()

if IS_JUPYTER:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from tqdm.utils import _term_move_up


def indexgrid(shape, device):
    return torch.stack(
        torch.meshgrid(
            [torch.arange(s, device=device) for s in shape],
            indexing="ij"
        )
    )

def set_trainable(model, trainable, verbose=False):
    for param in model.parameters():
        param.requires_grad = trainable

    if verbose:
        print_param_trainability(model)

def set_trainability(model, trainable, verbose=False):
    return set_trainable(model, trainable=trainable, verbose=verbose)


def freeze_bn(model, unfreeze=False, verbose=False):
    count = 0

    for name, child in model.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            if unfreeze:
                child.train()
            else:
                child.eval()

            count += 1

    if verbose:
        if unfreeze:
            print("Unfroze", count, "bn layers")
        else:
            print("Froze", count, "bn layers")


def print_param_trainability(model):
    trainable = 0
    untrainable = 0
    found_uninitialized = False

    for param in model.parameters():
        try:
            if param.requires_grad:
                trainable += np.prod(param.shape)
            else:
                untrainable += np.prod(param.shape)
        except RuntimeError:
            found_uninitialized = True

    print("Trainability for model", model._get_name())
    print("Trainable:", trainable)
    print("Untrainable:", untrainable)
    print("Total:", trainable + untrainable)
    if found_uninitialized:
        print("Some parameters were uninitialized and could therefore not be checked for trainability.")

def rescale(tensor, scale_factor, mode="nearest"):
    dt = tensor.dtype

    return torch.nn.functional.interpolate(
        tensor.reshape((
            tensor.shape[0],
            -1,
            tensor.shape[-2],
            tensor.shape[-1]
        )).to(torch.float32),
        (
            int(tensor.shape[-2] * scale_factor),
            int(tensor.shape[-1] * scale_factor)
        ),
        mode=mode
    ).squeeze().to(dt)


class EarlyStopping:
    def __init__(self, metric, patience, mode="max"):
        assert mode in ("min", "max")

        self.metric = metric
        self.patience = patience
        self.mode = mode
        self.best_val = None

        self.consumed_patience = 0

    def step(self):
        val = self.metric.compute()

        if self.best_val is None:
            self.best_val = val

            return False

        if self.mode == "min":
            if val < self.best_val:
                self.consumed_patience = 0
                self.best_val = val
            else:
                self.consumed_patience += 1
        elif self.mode == "max":
            if val > self.best_val:
                self.consumed_patience = 0
                self.best_val = val
            else:
                self.consumed_patience += 1
        else:
            raise ValueError()

        return self.consumed_patience >= self.patience


class TrainingObserver:
    class CycleManager:
        def __init__(self, observer, identifier, metrics, sparse_info=False):
            self.metrics = metrics
            self.start_time = None
            self.observer = observer
            self.identifier = identifier

            self.running_loss = 0
            self.counts = 0
            self.pbar = None
            self.additional_output_used = False
            self.sparse_info = sparse_info

            self.multi_output = len(metrics) != 0 and isinstance(metrics[0], (list, tuple))

            self._current_pbar_length = 0
            self._current_additional_term_ups = 0

            # Make metrics a list for uniform treatment
            if not self.multi_output:
                self.metrics = [self.metrics]

        def _assert_compatible(self, y_true, y_pred):
            if self.multi_output:
                assert all(
                    isinstance(item, (list, tuple)) for item in (y_true, y_pred)) and \
                    len(y_true) == len(y_pred) and \
                    len(y_true) == len(self.metrics), "The number of metric lists provided should match the number of target and true values"
            else:
                assert not isinstance(y_true, (list, tuple)) and \
                    not isinstance(y_pred, (list, tuple))

        def _init_pbar(self, max_val):
            term_cols = shutil.get_terminal_size((80, 20)).columns

            self._term_cols = term_cols

            self.pbar = tqdm(
                total=max_val,
                file=sys.stderr,
                position=0,
                leave=True,
                ascii=True,
                ncols=min(90, term_cols),
            )
            self.pbar.__enter__()

        def print_duration(self, batch_id, max_batches, do_update=True):
            if not self.pbar or batch_id == 0:
                self._init_pbar(max_batches)

            if do_update:
                # Disable progress bar updates except the last one in case of output redirects
                # if os.fstat(0) == os.fstat(1):
                #     if batch_id == max_batches - 1:
                #         self.pbar.update(batch_id)
                # else:
                self.pbar.update(1)

        def print_metrics(self, additional_outputs="", compute_expensive=False):
            metric_dict = self.eval_metrics(compute_expensive=compute_expensive)

            metric_strings = [f"Loss: {metric_dict['Loss']:.4f}"]

            for idx, metric_list in enumerate(self.metrics):
                for metric in metric_list:
                    metric_name = self.observer._get_metric_name(metric, idx)

                    metric_strings.append(f"{metric_name}: {metric_dict[metric_name]:.4f}")

            metric_string = ", ".join(metric_strings)
            output_str = "\n" + metric_string

            self._current_additional_term_ups = len(metric_string) // self._term_cols

            if additional_outputs:
                self._current_additional_term_ups += len(str(additional_outputs)) // self._term_cols

                output_str += "\n" + str(additional_outputs) + _term_move_up()

            if not IS_JUPYTER:
                final_output = output_str + _term_move_up() + _term_move_up() * self._current_additional_term_ups
            else:
                final_output = "; ".join(output_str.split("\n"))

            self.pbar.write(
                final_output,
                file=sys.stderr,
                end="",
            )

            self.additional_output_used = bool(additional_outputs)

        @torch.no_grad()
        def update(self, batch_id, n_batches, loss, y_pred, y_true, additional_outputs=""):
            if isinstance(y_true, (list, tuple)):
                self.counts += y_true[0].shape[0]
            else:
                self.counts += y_true.shape[0]

            self.running_loss += loss if not torch.is_tensor(loss) else loss.item()

            self._assert_compatible(y_pred, y_true)

            for i, metric_list in enumerate(self.metrics):
                for metric in metric_list:
                    metric(y_pred[i], y_true[i])

            if self.sparse_info:
                if batch_id == 0:
                    self._init_pbar(n_batches)
                if batch_id >= n_batches - 1:
                    self.pbar.update(n_batches)
                    self.print_metrics(
                        additional_outputs=additional_outputs,
                        compute_expensive=True,
                    )
            else:
                self.print_duration(batch_id, n_batches)
                self.print_metrics(
                    additional_outputs=additional_outputs,
                    compute_expensive=batch_id >= n_batches - 1,
                )

        def __enter__(self):
            self.start_time = time.time()

            for metric_list in self.metrics:
                for metric in metric_list:
                    metric.reset()

            print(f"{self.identifier}:")

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.pbar:
                self.pbar.__exit__(None, None, None)
                self.pbar.close()

            hist_dict = self.eval_metrics()
            self.observer.last_metrics = hist_dict

            for k, v in hist_dict.items():
                self.observer.history[self.identifier][k].append(v)

            print()
            if self.additional_output_used:
                print()

                for _ in range(self._current_additional_term_ups):
                    print()

        #@profile
        def eval_metrics(self, compute_expensive=False):
            metric_dict = {"Loss": self.running_loss / self.counts}

            for idx, metric_list in enumerate(self.metrics):
                for metric in metric_list:

                    if compute_expensive or \
                        not hasattr(metric, "_expensive") or \
                        not metric._expensive:

                        res = metric.compute()
                    else:
                        res = torch.nan

                    if torch.is_tensor(res) and res != torch.nan:
                        res = res.item()

                    metric_name =  self.observer._get_metric_name(metric, idx)

                    metric_dict[metric_name] = res

            return metric_dict

    def __init__(self, metrics=None, device=None, sparse_info=False):
        if not metrics:
            metrics = []

        self.multi_output = len(metrics) != 0 and isinstance(metrics[0], (list, tuple))

        # Make metrics a list for uniform treatment
        if not self.multi_output:
            metrics = [metrics]
        else:
            metrics = metrics

        if device is not None:
            for i in range(len(metrics)):
                for j in range(len(metrics[i])):
                    if hasattr(metrics[i][j], "to"):
                        metrics[i][j] = metrics[i][j].to(device)

        self.metrics = metrics
        self.history = None
        self.batch_id = None
        self.max_batches = None
        self.last_metrics = None
        self.sparse_info = sparse_info

        self._current_cycle_manager = None

    def _get_metric_name(self, metric, idx):
        if hasattr(metric, "name") and metric.name:
            metric_name = metric.name
        else:
            metric_name = metric.__class__.__name__

        if self.multi_output:
            return f"{metric_name}[{idx}]"
        else:
            return f"{metric_name}"

    def update(self, loss, y_pred, y_true, additional_outputs=""):
        assert self._current_cycle_manager is not None

        self._current_cycle_manager.update(
            self.batch_id,
            self.max_batches,
            loss,
            y_pred,
            y_true,
            additional_outputs=additional_outputs
        )

    def epoch(self, epoch_id, max_epochs):
        print(f"\nEpoch {epoch_id + 1}/{max_epochs}")

    def batch(self, batch_id, max_batches):
        self.batch_id = batch_id
        self.max_batches = max_batches

    def cycle(self, identifier):
        self.history[identifier] = {
            **{"Loss": list()},
            **{self._get_metric_name(metric, idx): list()
               for idx, metric_list in enumerate(self.metrics)
               for metric in metric_list}}

        self._current_cycle_manager = self.CycleManager(
            self,
            identifier,
            self.metrics,
            sparse_info=self.sparse_info,
        )

        return self._current_cycle_manager

    def __enter__(self):
        self.history = {}

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class ModuleOutputCache:
    def __init__(self, max_cache_size=1e-8, subsample_count=1000) -> None:
        self.cache = {}
        self.max_cache_size = max_cache_size

        # Method of hashing is based on torchcache https://github.com/meakbiyik/torchcache/blob/main/torchcache/torchcache.py
        roll_powers = torch.arange(0, subsample_count * 2) % 15
        self.subsample_count = subsample_count
        self.coefficients = (
            torch.pow(torch.tensor([2.0]), roll_powers).float().detach().view(1, -1)
        )
        self.coefficients.requires_grad_(False)

    def clear_cache(self):
        self.cache = {}

    def cache(self, model, inputs, outputs):
        if not isinstance(inputs, tuple):
            inputs = (inputs,)

        param_hashes = [self._hash_tensor(v) for v in model.parameters()]
        input_hashes = [self.hash_item(v) for v in model.parameters()]

        v = "".join(param_hashes) + "".join(input_hashes)

        hash = hashlib.md5(v.encode()).hexdigest()

        self.cache[hash] = outputs

    def _hash_item(self, item):
        if isinstance(item, (dict, tuple, list)):
            return hashlib.md5(str(item).encode()).hexdigest()
        else:
            return self._hash_tensor(item)

    def _hash_tensor(self, tensor):
        self.coefficients = self.coefficients.to(tensor.device)

        subsample_rate = max(1, tensor.shape[1] // self.subsample_count)
        tensor = tensor[:, ::subsample_rate]

        # in mixed precision, the matmul operation returns float16 values,
        # which overflows
        hash_val = torch.sum(
            tensor * self.coefficients[:, :tensor.shape[1]],
            dim=1,
            dtype=torch.long,
        )

        return str(hash_val.item())

