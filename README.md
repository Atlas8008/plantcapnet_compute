# PlantCAPNet Backend

The backend of the PlantCAPNet framework. It supports zeroshot training and training with custom cover data, which can be initiated using the `construct_ensemble.py` script.


## Data

The system searches for datasets in the subfolder `datasets/`, which might be needed to be created first. In this folder, all custom datasets can be put, or existing datasets from other paths can be symlinked to that directory (`ln -s`).

For datasets with corresponding cover and phenology annotations, bring the dataset into the structure described in [`datasets/dataset_structure.md`](datasets/dataset_structure.md) for the system to be able to read it. To check, if the dataset structure is correct, run the `tools/check_dataset_structure.py` script, which analyses the structure and notifies about any obvious deviations from the supposed structure.

## Setup

For setup we recommend a working installation of anaconda for python package management. Furthermore, the backend requires python 3.10 or newer.

Install the python requirements via pip:

```
pip install -r requirements.txt
```

For full functionality, we recommend creating a new virtual environment with [anaconda](https://www.anaconda.com/download) and install all requirements in this environment using the `install_env.sh` script:

```
source install_env.sh
```

Note that the script should be run with `source` instead of `bash` or `sh`. Running with either of these might cause issues during the installation.

The script will create a new python environment with an R installation to support functionalities like DCA and Procrustes Tests from the R package `vegan`. It will also install all requirements for python. It should be noted that the environment installed via the installation script of the web frontend can also be used for running the backend scripts.


## Running the Training

Running the training can be done via
```
python construct_ensemble.py <args>
```

For a list of usable arguments, run `python construct_ensemble.py -h` and `python run_training_pipeline.py -h`, as the former passes down arguments to the seconds script. Depending on the setup, the `construct_ensemble.py` script can run the training for multiple models locally or alternatively on a [SLURM](https://slurm.schedmd.com/documentation.html) cluster, to enable highly parallel and high performance training.

We recommend configuring training with the PlantCAPNet web frontend.

The training can take some time, depending on the size of the datasets, number of classes (species), number of epochs trained and image resolutions used. For example, using an image dataset with 100.000 images with cover and phenology annotations with image resolutions of about 3200x1600 px and 40 epochs, the plant cover and phenology training alone can take 3-5 days on a NVIDIA A100 GPU. The duration of the pre-training is primarily dependent on the number of images and the number of classes. With about 75 classes and 600 training images per class, the entire pre-training process can take 2-4 days. The training durations scale approximately linearly, so this might help estimating the entire duration of a training process.

After successful training, the script will copy the trained model(s) to [`output_models/`](output_models/) into a subfolder corresponding to your model run name.

## Checking Performance

To check, how well the trained model(s) perform on the validation data, check the [outputs/](outputs/) folder. Specifically, to the the performance of your zero-shot or cover-trained model, go to either the `zeroshot/` or `cover/` subfolder, and then the subfolder with your model run name. There you can find several files containing different aspects/metrics analyzed during evaluation, helping you in making a decision about the best model to use for inference.

## Recommended Hardware

Hardware requirements vary based on the intended use (inference or training). For inference tasks, a CUDA-compatible NVIDIA GPU possessing at least 8 GB of VRAM (e.g., NVIDIA GeForce RTX 2080 or equivalent) is recommended to achieve reasonable performance. Model training, being more computationally intensive, benefits significantly from a CUDA-compatible NVIDIA GPU with 12 GB of VRAM or more (e.g., NVIDIA GeForce RTX 3080, RTX 4070 Ti, or data center-grade GPUs like A100). Regardless of the primary task, a minimum of 64 GB of system memory (RAM) is recommended.

## Testing the System

There are two scripts for testing the functionality of the system, both of which are located under `scripts/`. To run them, you require a pre-training dataset obtainable with the GBIF-downloader. With the downloader, download the provided example dataset `example_ds`, perform a training and validation split and copy or move it into the `datasets/` folder. For testing the cover and phenology training functionality, a small test dataset is provided under `datasets/InsectArmageddon_test/`, which can also serve as example dataset for potential custom datasets.

To test the plant cover and phenology training functionality, run
```
bash test_script_cover_iatest.sh
```

This will pre-train first on the pre-training dataset once with classification pre-training and once with segmentation pre-training, and then train an ensemble of 2 models each trained with 10, 20 and 30 epochs on the `InsectArmageddon_test` dataset. After the training, the models and the generated ensemble can be found under `output_models/test_model_cover`.

To test the zero-shot training functionality, obtain the pre-training dataset as described above, and run
```
bash test_script_zeroshot_iatest.sh
```

This will also train an ensemble of two repetitions of ConvNext_tiny, ConvNext_base and ConvNext_large, and compile them into an ensemble model after training. These models can be found afterwards under `output_models/test_model_zeroshot`.

It should be noted that it can happen that the test crashes due to insufficient GPU memory (CUDA OutOfMemoryError). In this case, try decreasing the batch size (especially for segmentation pre-training), i.e., lower `c_batch_size`or `s_batch_size` in the test scripts.

## Cleanup

During the training process, the models, parameters, outputs and metrics are saved. To prevent this data from taking too much space, it is recommended to occasionally clean or delete the folders...
- `meta/`: ...containing experimental metadata and references to models
- `saved_models/`: ...containing intermediate and final models generated during training
- `outputs/`: ...containing metrics and output data of the models
- `output_models/`: ...containing copies of the final output models for easy access