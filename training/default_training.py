import torch

from utils.torch_utils.multiio import *


def _maybe_format_loss_display(losses):
    if len(losses) == 1:
        return ""
    else:
        return ", ".join(f"Loss[{i}]: {losses[i].item():.4f}" for i in range(len(losses)))


def get_losses(y, pred, criterion, mask):
    losses = []

    for single_criterion, single_pred, single_y in zip(criterion, pred, y):
        if mask is None:
            loss_ = single_criterion(single_pred, single_y)
        else:
            loss_ = single_criterion(single_pred, single_y, mask=mask)

        losses.append(loss_)

    return losses


def train(*, model, n_epochs, observer, criterion, device, training_loader, validation_loader, optimizer, scheduler, scheduler_metric=None, log_data=None, early_stopping=None, force_validation=False, gradient_clip_val=None, **kwargs):
    """Train the model for a number of epochs.

    Args:
        model (torch.nn.Module): The model to train.
        n_epochs (int): The number of epochs to train for.
        observer (Observer): The observer to use for logging.
        criterion (torch.nn.Module): The loss function to use.
        device (torch.device): The device to train on.
        training_loader (torch.utils.data.DataLoader): The training data loader.
        validation_loader (torch.utils.data.DataLoader): The validation data loader.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler to use. Defaults to None.
        scheduler_metric (Metric, optional): The metric to use for the scheduler. Defaults to None.
        log_data (list, optional): A list to log data into. Defaults to None.
        early_stopping (EarlyStopping, optional): An early stopping object. Defaults to None.
        force_validation (bool, optional): Whether to force validation at the end of training. Defaults to False.
        gradient_clip_val (float, optional): The value for gradient clipping. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the model.
    """
    scaler = torch.amp.GradScaler(device=device.type)

    validation_metrics = None

    ep = None

    for ep in range(n_epochs):
        observer.epoch(ep, n_epochs)

        model.train()

        with observer.cycle("training"):
            for batch_id, t in enumerate(training_loader):
                observer.batch(batch_id, len(training_loader))

                x, y, mask = _unpack_tuple(t, device=device)

                pred = _predict(x, model, **kwargs)

                criterion = _to_list(criterion)
                _assert_compatible(y, pred, criterion)

                losses = get_losses(y, pred, criterion, mask=mask)
                loss = torch.sum(torch.stack(losses))

                optimizer.zero_grad()
                scaler.scale(loss).backward()

                if gradient_clip_val is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip_val)

                scaler.step(optimizer)
                scaler.update()

                additional_displays = []

                if loss_display := _maybe_format_loss_display(losses):
                    additional_displays.append(loss_display)

                lr = [group['lr'] for group in optimizer.param_groups]

                additional_displays.append(f"LR: {lr}")

                if hasattr(model, "regularization_losses"):
                    reg_losses = model.regularization_losses()
                    loss += reg_losses
                    additional_displays.append(f"RegLoss: {reg_losses.item():.4f}")

                observer.update(
                    loss,
                    pred,
                    y,
                    ", ".join(additional_displays)
                )

        validation_metrics = evaluate(model, device, validation_loader, criterion, observer, **kwargs)

        if scheduler is not None:
            if scheduler_metric is None:
                scheduler.step()
            else:
                scheduler.step(scheduler_metric.compute())

        if early_stopping is not None and early_stopping.step():
            print("Early stopping.")
            break

    if validation_metrics is None and force_validation:
        validation_metrics = evaluate(
            model,
            device,
            validation_loader,
            criterion,
            observer,
            **kwargs
        )

    if log_data is not None:
        log_data.append(validation_metrics)

    if ep is not None:
        validation_metrics["epochs_trained"] = ep + 1

    return validation_metrics


def evaluate(model, device, validation_loader, criterion, observer, term="validation", **kwargs):
    """Evaluate the model.

    Args:
        model (torch.nn.Module): The model to evaluate.
        device (torch.device): The device to evaluate on.
        validation_loader (torch.utils.data.DataLoader): The validation data loader.
        criterion (torch.nn.Module or list): The loss function(s) to use.
        observer (Observer): The observer to use for logging.
        term (str, optional): The term to use for logging. Defaults to "validation".
    Returns:
        dict: The last metrics from the observer.
    """
    obs = observer

    model.eval()

    with obs.cycle(term), torch.no_grad():
        for batch_id, t in enumerate(validation_loader):
            obs.batch(batch_id, len(validation_loader))

            x, y, mask = _unpack_tuple(t, device=device)

            pred = _predict(x, model, **kwargs)

            criterion = _to_list(criterion)
            _assert_compatible(y, pred, criterion)

            losses = get_losses(y, pred, criterion, mask=mask)
            loss = torch.sum(torch.stack(losses))

            additional_displays = []

            if loss_display := _maybe_format_loss_display(losses):
                additional_displays.append(loss_display)

            if hasattr(model, "regularization_losses"):
                reg_losses = model.regularization_losses()
                loss += reg_losses
                additional_displays.append(f"RegLoss: {reg_losses.item():.4f}")

            obs.update(loss, pred, y, ", ".join(additional_displays))

    return obs.last_metrics