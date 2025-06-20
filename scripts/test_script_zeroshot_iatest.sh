#! /bin/bash

cd ..

python construct_ensemble.py \
	--name test_model_zeroshot \
	--slurm False \
	--slurm_partition gpu \
	--pt_dataset_name example_ds \
	--community_dataset_name none \
	--run_evaluation False \
	--c_learning_rate 5e-05 \
	--c_weight_decay 1e-05 \
	--c_n_epochs -1 \
	--c_batch_size 4 \
	--c_image_size 448 \
	--c_use_occlusion_augmentation False \
	--c_threshold 0.2 \
	--c_normalization torch \
	--c_enriched_wsol_output True \
	--c_loss cce \
	--cm_pooling_method lse \
	--wf_base_network efficientnet_v2_l \
	--wf_fpn_spec P1-128 \
	--s_learning_rate 1e-05 \
	--s_weight_decay 1e-05 \
	--s_n_epochs 3 \
	--s_batch_size 3 \
	--s_image_size 448 \
	--s_normalization torch \
	--s_use_cutout True \
	--s_ic_min 32 \
	--s_ic_max 224 \
	--s_ic_reps 2 \
	--s_image_caching True \
	--s_max_workers 16 \
	--s_loss bce_dice \
	--sf_base_network convnext_large \
	--sf_fpn_spec P1-128 \
	--use_ensemble True \
	--ensemble_models convnext_tiny convnext_base convnext_large \
	--n_ensemble_reps 2 \
	--zeroshot True
