import os
import torch
import platform

import torch.distributed

print("Running on host", platform.node())

from torch import nn

import models

from training.default_training import train
from training import distributed as dist

from cap_utils import get_arg_attribute, eval_and_save
from utils.torch_utils import TrainingObserver, print_param_trainability

import training_configurations as tc
import arguments

def sortbykey(d):
    return {k: d[k] for k in sorted(d.keys())}

def reset_params(m):
    if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
        m.reset_parameters()

def get_restore_config(configurations, restore_checkpoint, config_idx):
    if isinstance(configurations[restore_checkpoint], tc.MetaConfiguration):
        return configurations[restore_checkpoint][config_idx]
    else:
        return configurations[restore_checkpoint]

def run_training(rank, world_size, mode, n_outputs_pretraining, n_outputs_target_task, restore_checkpoint, args, config, config_idx, configurations):
    is_distributed = args["main"].multigpu

    print("##### TRAINING: ", mode.upper(), "#####")
    if is_distributed:
        dist.setup_distributed(rank, world_size)
        device = torch.device("cuda:" + str(rank))
    else:
        device = torch.device("cuda:" + args["main"].gpu)

    model = models.build_model(
        mode,
        n_outputs_pretraining=n_outputs_pretraining,
        n_outputs_target_task=n_outputs_target_task,
        args=args,
    )

    model = model.to(device)

    model = config.modify_model_with_configs(
        model,
        configurations,
        device
    )
    model, is_ensemble = config.maybe_build_ensemble(
        model,
        configurations,
        device,
    )

    if config.args.dataset.lower() == "none":
        if not is_ensemble:
            # Do a dry run and restore model parameters if needed
            with torch.no_grad():
                model.eval()
                inp = torch.zeros((1, 3, 512, 512)).to(device)

                model(
                    inp,
                    **config.model_kwargs,
                )

            if restore_checkpoint is not None:
                if restore_checkpoint == "reset":
                    model.apply(reset_params)
                else:
                    restore_config = get_restore_config(
                        configurations=configurations,
                        restore_checkpoint=restore_checkpoint,
                        config_idx=config_idx
                    )

                    restore_config.restore_model_from_checkpoint(
                        model
                    )

            config.set_model_trainability(model)
    else: # Proceed with training
        training_loader = config.training_loader
        validation_loader = config.eval_loader

        if is_distributed:
            training_loader = dist.distributed_data_loader(training_loader)
            validation_loader = dist.distributed_data_loader(validation_loader)

        if not is_ensemble:
            with torch.no_grad():
                model.eval()
                inp = next(iter(validation_loader))[0]

                if isinstance(inp, dict):
                    inp = {k: v.to(device) for k, v in inp.items()}
                elif isinstance(inp, (tuple, list)):
                    inp = [it.to(device) for it in inp]
                else:
                    inp = inp.to(device)

                model(
                    inp,
                    **config.model_kwargs,
                )

            if restore_checkpoint is not None:
                if restore_checkpoint == "reset":
                    model.apply(reset_params)
                else:
                    restore_config = get_restore_config(
                        configurations=configurations,
                        restore_checkpoint=restore_checkpoint,
                        config_idx=config_idx
                    )

                    restore_config.restore_model_from_checkpoint(
                        model
                    )

            config.set_model_trainability(model)

        if is_distributed:
            torch.distributed.barrier()
            for param in model.parameters():
                torch.distributed.broadcast(param.data, src=0)

            model = models.DistributedParallelWrapper(model, device_ids=[rank], find_unused_parameters=True)

        print(model)
        print_param_trainability(model)

        optimizer = config.get_optimizer(model)
        scheduler = config.get_scheduler(optimizer)

        if hasattr(config.args, "gradient_clip"):
            gradient_clip_val = config.args.gradient_clip
        else:
            gradient_clip_val = None

        print("Early stopping:", config.early_stopping)
        print("Training for", config.epochs, "epochs")

        with TrainingObserver(metrics=config.metrics, device=device, sparse_info=args["main"].training_sparse_info) as obs:
            val_metrics = train(
                model=model,
                n_epochs=config.epochs,
                observer=obs,
                criterion=config.loss,
                device=device,
                training_loader=training_loader,
                validation_loader=validation_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scheduler_metric=config.scheduler_metric,
                early_stopping=config.early_stopping,
                force_validation=True,
                gradient_clip_val=gradient_clip_val,
                **config.model_kwargs,
            )

    if is_distributed:
        # If model is wrapped in DataParallel, extract the original model
        model = model.module

        if rank != 0:
            torch.distributed.destroy_process_group()
            return

    if config.model_save_path:
        os.makedirs(
            os.path.dirname(config.model_save_path),
            exist_ok=True,
        )
        os.makedirs(
            os.path.dirname(config.model_link_path),
            exist_ok=True,
        )

        print("Saving model to", config.model_save_path)
        torch.save(model, config.model_save_path)

        if not os.path.exists(config.model_link_path) and config.keep_saved_model:
            os.symlink(
                os.path.relpath(
                    config.model_save_path,
                    os.path.dirname(config.model_link_path),
                ),
                config.model_link_path,
            )

    config.save_hash_and_config()

    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = arguments.get_arguments()

    device = torch.device("cuda:" + args["main"].gpu)

    configurations = {}

    configurations.update(
        tc.get_classification_configurations(
            experiment_name=args["main"].experiment_name,
            args=args["classification"],
        )
    )

    # Copy any desired parameters for segmentation setups from the wsol setup (marked with "__wsol__")
    for argname in ["segmentation", "deocclusion", "joint_deocclusion"]:
        for k, v in vars(args[argname]).items():
            if v == "__wsol__":
                setattr(args[argname], k, getattr(args["classification"], k))


    # Update wsol meta info for correct segmentation path in segmentation configs
    configurations["wsol"].add_meta_info(tag=args["main"].tag)
    configurations["wsol"].add_info(
        extractor_params=sortbykey(vars(args["wsol_fe_model"])))
    configurations["wsol"].add_info(
        module_params=sortbykey(vars(args["classification_model"])))

    configurations.update(
        tc.get_segmentation_configurations(
            experiment_name=args["main"].experiment_name,
            segmentation_args=args["segmentation"],
            deocclusion_args=args["deocclusion"],
            joint_args=args["joint_deocclusion"],
            segmentation_path=configurations["wsol"].output_path,
        )
    )
    configurations["segmentation"].add_info(
        extractor_params=sortbykey(vars(args["segmentation_fe_model"])))
    configurations["segmentation"].add_info(
        module_params=sortbykey(vars(args["segm_model"])))
    configurations["segmentation"].add_info(
        wsol_base_hash=configurations["wsol"].arg_hash)
    configurations["deocclusion"].add_info(
        module_params=sortbykey(vars(args["deoc_model"])))
    configurations["deocclusion"].add_info(
        wsol_base_hash=configurations["wsol"].arg_hash)

    configurations.update(
        tc.get_cover_configurations(
            experiment_name=args["main"].experiment_name,
            zeroshot_args=args["zeroshot_cover"],
            cover_args=args["cover_training"],
            temporal_cover_args=args["temporal_cover_training"],
            cover_eval_args=args["cover_eval"],
        )
    )
    configurations["zeroshot"].add_info(
        module_params=sortbykey(vars(args["zeroshot_module"])))
    configurations["cover"].add_info(
        module_params=sortbykey(vars(args["cover_model"])))
    configurations["temporal_cover"].add_info(
        module_params=sortbykey(vars(args["temporal_cover_model"])))

    n_outputs_pretraining = configurations["wsol"].n_model_outputs
    n_outputs_target_task = configurations["cover"].n_model_outputs

    print("Segmentation path:", configurations["wsol"].output_path)
    print("Classification dataset:", configurations["segmentation"].args.dataset)
    print("Segmentation dataset:", configurations["segmentation"].args.dataset)
    print("# Outputs Pre-Training:", n_outputs_pretraining)
    print("# Outputs Target Task:", n_outputs_target_task)

    #configurations["segmentation"].save_hash_and_config()

    pipeline_modes = [
        "wsol",
        "segmentation",
        "deocclusion",
        "joint_deocclusion",
        "zeroshot",
        "cover",
        "temporal_cover",
    ]

    for mode in pipeline_modes:
        _config = configurations[mode]

        do_train = get_arg_attribute(
            args["main"],
            mode,
            "training",
        )
        run_eval = get_arg_attribute(
            args["main"],
            mode,
            "evaluation",
        )
        run_inference = get_arg_attribute(
            args["main"],
            mode,
            "inference",
        )

        if run_eval is None:
            run_eval = do_train
        if run_inference is None:
            run_inference = do_train

        if isinstance(_config, tc.MetaConfiguration):
            _configs = _config.configurations
        else:
            _configs = [_config]

        for config_idx, config in enumerate(_configs):
            restore_checkpoint = get_arg_attribute(
                args["main"],
                mode,
                "restore_checkpoint",
            )

            if restore_checkpoint is not None and restore_checkpoint.lower() == "none":
                restore_checkpoint = None

            config.add_meta_info(tag=args["main"].tag)

            # Add restore checkpoint file path to additional info for config
            # to ensure accurate config hash
            if restore_checkpoint is not None:
                if restore_checkpoint == "reset":
                    restore_data = "reset"
                    print("Resetting weights for model")
                else:
                    restore_config = get_restore_config(
                        configurations=configurations,
                        restore_checkpoint=restore_checkpoint,
                        config_idx=config_idx
                    )
                    restore_data = restore_config.model_save_path
                    print("Restoring model from", restore_data)

                config.add_info(
                    restore_checkpoint=restore_data,
                )

            if do_train and (
                not args["main"].only_train_nonexisting or
                not os.path.isfile(config.model_save_path)
                ):

                if args["main"].multigpu:
                    world_size = torch.cuda.device_count()

                    print("World size:", world_size)

                    torch.multiprocessing.spawn(
                        run_training,
                        args=(world_size, mode, n_outputs_pretraining, n_outputs_target_task, restore_checkpoint, args, config, config_idx, configurations),
                        nprocs=world_size,
                        join=True,
                    )
                else:
                    run_training(0, 1, mode, n_outputs_pretraining, n_outputs_target_task, restore_checkpoint, args, config, config_idx, configurations)

            if run_eval or run_inference: # Evaluate and log metrics
                print("Loading model from", config.model_save_path)
                model = torch.load(config.model_save_path, map_location=device, weights_only=False)

                print("Eval/inference model hash:", hash(str(list(model.parameters()))))

                if run_eval:
                    print("##### EVALUATION: ", mode.upper(), "#####")

                    evaluation_methods = config.evaluation_methods

                    for evaluation_method in evaluation_methods:
                        eval_and_save(
                            evaluation_method,
                            config,
                            model=model,
                            device=device,
                        )

                if run_inference:
                    print("##### INFERENCE: ", mode.upper(), "#####")

                    for inference_method in config.inference_methods:
                        inference_method(model, device)

            config.clear_cache()

            if not config.keep_saved_model and os.path.exists(config.model_save_path):
                print("Removing model from", config.model_save_path)
                os.remove(config.model_save_path)

        if do_train and isinstance(_config, tc.MetaConfiguration):
            if run_eval:
                evaluation_methods = _config.post_evaluation_methods
                for evaluation_method in evaluation_methods:
                    eval_and_save(
                        evaluation_method,
                        _config,
                    )

            if run_inference:
                for inference_method in _config.post_inference_methods:
                    inference_method()