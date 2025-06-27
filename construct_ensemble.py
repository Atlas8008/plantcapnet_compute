import os
import sys
import shutil
import submitit
import argparse
import subprocess


def str2bool(s):
    if s.lower() in ("1", "true", "yes"):
        return True
    elif s.lower() in ("0", "false", "no"):
        return False
    else:
        raise ValueError("Invalid boolean value: " + s)

def launch_process(cmd, copy_params=None):
    p = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    p.communicate()

    if copy_params:
        copy_model(
            **copy_params,
        )

    return p

def copy_model(run_name, exp_name, mode, repetition=None):
    assert mode in ("cover", "zeroshot")

    directory = os.path.join("output_models", run_name)

    os.makedirs(directory, exist_ok=True)

    rep_str = ""
    rep_underscore = ""

    if repetition is not None:
        rep_str = f"rep{repetition}"
        rep_underscore = "_"

    src = os.path.join("meta", "model_links", mode, exp_name + f"val_{rep_str}.pth")
    dst = os.path.join(directory, exp_name + f"{rep_underscore}{rep_str}.pth")

    print("Copying model from", src, "to", dst)
    shutil.copyfile(src, dst)


def add_or_replace_arg(key, val, lst):
    if key in lst:
        idx = lst.index(key)
        lst[idx + 1] = val
    else:
        lst.append(key)
        lst.append(val)


def launch_job(cmd, name, run_on_slurm, partition=None, dependencies=None, copy_params=None, ngpus=1, ncpus=10, mem_per_gpu=57, addtional_slurm_params=None):
    if run_on_slurm:
        optional_args = {}

        if partition is not None:
            optional_args["slurm_partition"] = partition
        if dependencies is not None:
            optional_args["slurm_dependency"] = dependencies if isinstance(dependencies, str) else ",".join(map(str, dependencies))

        if addtional_slurm_params is not None:
            optional_args.update(addtional_slurm_params)

        executor = submitit.AutoExecutor(folder="logdir")
        executor.update_parameters(
            name=name,
            mem_gb=mem_per_gpu * ngpus,
            cpus_per_task=ncpus,
            nodes=1,
            timeout_min=8 * 24 * 60,
            #gres="gpu:1",
            gpus_per_node=ngpus,
            **optional_args
        )

        job = executor.submit(launch_process, cmd, copy_params)
    else:
        launch_process(cmd, copy_params)

        job = None

    return job


parser = argparse.ArgumentParser(description="This script constructs the training for models and converts them into an ensemble.")

parser.add_argument("--name", type=str, help="The name of the model training run.")
parser.add_argument("--only_train_nonexisting", type=str2bool, default=False, help="Flag; if set, only models will be trained that do not exist already.")
parser.add_argument("--ensemble_epochs", nargs="*", type=str, default=None, help="The epoch counts for the ensemble submodels.")
parser.add_argument("--n_ensemble_reps", type=int, default=1, help="The number of identical repetition for the ensemble submodels.")
parser.add_argument("--ensemble_models", nargs="*", type=str, default=None, help="The names of the architectures to use as ensemble submodels.")
parser.add_argument("--run_evaluation", type=str2bool, default=True, help="Flag; if set, the model will be evaluated after training. If not set, the model is only trained.")

parser.add_argument("--use_ensemble", type=str2bool, help="Flag; if set, the model will be trained as an ensemble.")
parser.add_argument("--zeroshot", type=str2bool, help="Flag; if set, the model will be trained for zero-shot application, otherwise full cover prediction training.")
parser.add_argument("--skip_wsol", type=str2bool, help="Flag; if set, the classification training and inference phase will be skipped. Can be used to prevent unnecessary training and inference, if the segmentations already exist.")

parser.add_argument("--pt_dataset_name", type=str, help="The name or path of the pre-training dataset to use.")
parser.add_argument("--community_dataset_name", type=str, help="The name or path of the plant community dataset to use.")

parser.add_argument("--gpu", type=str, help="The ID of the GPU to use for training.")
parser.add_argument("--ngpus", type=int, default=1, help="The number of GPUs to use for training.")
parser.add_argument("--ncpus", type=int, default=10, help="The number of CPUs to use for training.")
parser.add_argument("--slurm", type=str2bool, help="Flag; if set, the training will be submitted to the SLURM queue.")
parser.add_argument("--slurm_partition", type=str, help="The partition to submit the job to.")
parser.add_argument("--slurm_mem_gb_per_gpu", type=int, default=57, help="The memory per GPU to request via slurm in GB.")
parser.add_argument("--slurm_params", type=str, nargs="*", default="", help="Additional parameters to pass to the SLURM job submission. Denoted like <key>=<value>.Multiple parameters should be separated by spaces.")



args, remaining_params = parser.parse_known_args()

slurm_params = {}

if args.slurm_params:
    for param in args.slurm_params:
        key, value = param.split("=")
        slurm_params[key] = value

if args.zeroshot:
    args.ensemble_epochs = None

if args.ensemble_epochs is None:
    model_epochs = [None]
else:
    model_epochs = args.ensemble_epochs
    args.ensemble_epochs = args.ensemble_epochs or [args.ensemble_epochs]

if args.ensemble_models is None:
    args.ensemble_models = ["none"]

args.ensemble_models = args.ensemble_models or [args.ensemble_models]

exp_names = []
jobs = []

additional_params = []

if args.zeroshot:
    additional_params.append("--cp_image_size")
    additional_params.append("1536")

if args.ngpus > 1:
    additional_params.append("--m_multigpu")
    additional_params.append("True")

for epochs in model_epochs:
    for model in args.ensemble_models:
        exp_name = f"{args.name}_model{model}_epochs{epochs}"

        for rep in range(args.n_ensemble_reps):
            if epochs is not None:
                if not args.zeroshot:
                    # Replace model epochs with ensemble epochs
                    add_or_replace_arg(
                        "--cp_n_epochs",
                        epochs,
                        remaining_params,
                    )

            if model is not None:
                add_or_replace_arg(
                    "--sf_base_network",
                    model,
                    remaining_params
                )

            cmd = \
                ["python"] + [f"run_training_pipeline.py"] + \
                ["--m_gpu"] + [f"0"] + \
                ["--m_wsol_training"] + [str(not args.skip_wsol)] + \
                ["--m_segmentation_training"] + [f"True"] + \
                ["--m_zeroshot_training"] + [f"{args.zeroshot}"] + \
                ["--m_zeroshot_restore_checkpoint"] + [f"segmentation"] + \
                ["--m_cover_training"] + [f"{not args.zeroshot}"] + \
                ["--m_cover_restore_checkpoint"] + [f"segmentation"] + \
                ["--m_training_sparse_info"] + [f"True"] + \
                ["--m_only_train_nonexisting"] + [f"{args.only_train_nonexisting}"] + \
                ["--m_cover_evaluation"] + [f"{not args.zeroshot and args.run_evaluation}"] + \
                ["--m_cover_inference"] + [f"False"] + \
                ["--m_zeroshot_evaluation"] + [f"{args.zeroshot and args.run_evaluation}"] + \
                ["--m_zeroshot_inference"] + [f"False"] + \
                ["--m_tag"] + [f"rep{rep}"] + \
                ["--z_keep_saved_model"] + [f"True"] + \
                ["--cp_keep_saved_model"] + [f"True"] + \
                ["--z_image_size"] + [f"1536"] + \
                ["--z_inference_configurations"] + [f"bzero,dnone"] + \
                ["--cpm_pheno_model"] + [f"default"] + \
                ["--cpm_pheno_prediction_mode"] + [f"jointad3"] + \
                ["--c_dataset"] + [args.pt_dataset_name] + \
                ["--s_dataset"] + [args.pt_dataset_name] + \
                ["--cp_dataset"] + [args.community_dataset_name] + \
                ["--z_dataset"] + [args.community_dataset_name] + \
                ["--m_experiment_name"] + [exp_name]

            cmd += remaining_params + additional_params

            if not args.slurm:
                print(f"Launched job for {exp_name} with command: {cmd}")

            job = launch_job(
                cmd,
                exp_name,
                args.slurm,
                args.slurm_partition,
                ngpus=args.ngpus,
                ncpus=args.ncpus,
                mem_per_gpu=args.slurm_mem_gb_per_gpu,
                copy_params={
                    "run_name": args.name,
                    "exp_name": exp_name,
                    "mode": "zeroshot" if args.zeroshot else "cover",
                    "repetition": rep,
                },
                addtional_slurm_params=slurm_params,
            )

            jobs.append(job)

            if args.slurm and job is not None:
                print(f"Launched job {job.job_id} for {exp_name} with command: {cmd}")

        exp_names.append(exp_name)



if args.use_ensemble:
    exp_name = f"{args.name}_ensemble"

    if args.ensemble_epochs is not None:
        ensemble_args = ["--cp_ensemble_epochs"] + args.ensemble_epochs
    else:
        ensemble_args = []

    if args.zeroshot:
        ensemble_args += ["--z_ensemble_model_names"] + exp_names
        ensemble_args += ["--z_ensemble_meta_tag_suffixes"] + [f'rep{r}' for r in range(args.n_ensemble_reps)]
    else:
        ensemble_args += ["--cp_ensemble_model_names"] + exp_names
        ensemble_args += ["--cp_ensemble_meta_tag_suffixes"] + [f'rep{r}' for r in range(args.n_ensemble_reps)]


    add_or_replace_arg(
        "--cp_n_epochs",
        "0",
        remaining_params,
    )

    ensemble_cmd = \
        ["python"] + [f"run_training_pipeline.py"] + \
        ["--m_gpu"] + [f"0"] + \
        ["--m_wsol_training"] + [f"False"] + \
        ["--m_segmentation_training"] + [f"False"] + \
        ["--m_deocclusion_training"] + [f"False"] + \
        ["--m_joint_deocclusion_training"] + [f"False"] + \
        ["--m_zeroshot_training"] + [f"{args.zeroshot}"] + \
        ["--m_zeroshot_restore_checkpoint"] + [f"none"] + \
        ["--m_cover_training"] + [f"{not args.zeroshot}"] + \
        ["--m_cover_restore_checkpoint"] + [f"none"] + \
        ["--m_training_sparse_info"] + [f"True"] + \
        ["--m_only_train_nonexisting"] + [f"False"] + \
        ["--m_cover_evaluation"] + [f"{not args.zeroshot and args.run_evaluation}"] + \
        ["--m_cover_inference"] + [f"False"] + \
        ["--m_zeroshot_evaluation"] + [f"{args.zeroshot and args.run_evaluation}"] + \
        ["--m_zeroshot_inference"] + [f"False"] + \
        ["--z_enriched_eval"] + [f"False"] + \
        ["--z_inference_configurations"] + [f"bzero,dnone"] + \
        ["--z_keep_saved_model"] + [f"True"] + \
        ["--z_image_size"] + [f"1536"] + \
        ["--cp_keep_saved_model"] + [f"True"] + \
        ["--cpm_pheno_prediction_mode"] + [f"jointad3"] + \
        ensemble_args + \
        ["--c_dataset"] + [args.pt_dataset_name] + \
        ["--s_dataset"] + [args.pt_dataset_name] + \
        ["--cp_dataset"] + [args.community_dataset_name] + \
        ["--z_dataset"] + [args.community_dataset_name] + \
        ["--m_experiment_name"] + [exp_name]

    ensemble_cmd += remaining_params + additional_params

    print(ensemble_cmd)

    job = launch_job(
        ensemble_cmd,
        exp_name,
        args.slurm,
        args.slurm_partition,
        dependencies=[j.job_id for j in jobs] if args.slurm else None,
        ngpus=args.ngpus,
        ncpus=args.ncpus,
        mem_per_gpu=args.slurm_mem_gb_per_gpu,
        copy_params={
            "run_name": args.name,
            "exp_name": exp_name,
            "mode": "zeroshot" if args.zeroshot else "cover"
        },
        addtional_slurm_params=slurm_params,
    )

    if args.slurm and job is not None:
        print(f"Launched job {job.job_id} for {exp_name} with command: {ensemble_cmd}")
    else:
        print(f"Launched job for {exp_name} with command: {ensemble_cmd}")

