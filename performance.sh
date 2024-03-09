export TORCH_HOME=$(pwd) &&
export PYTHONPATH="$(pwd)/src/lib/FcF:$(pwd)/src/lib/MAT:"
python performance.py \
--experiment_name mobilefill_celeba-hq \
--dataset_name Celeba-hq \
--model_names MobileFill_WO_MOBILEVIT,MAT \
--mask_type thick_256 \
--target_size 256 \
--total_num 10000 \
--max_vis_num 500 \
--sample_num 5 \
--img_dir "./dataset/validation/Celeba-hq/thick_256/imgs" \
--mask_dir "./dataset/validation/Celeba-hq/thick_256/masks" \
--save_dir ./results \
--aspect_ratio_kept \
--fixed_size \
--center_crop \
--batch_size 10 \
--get_diversity


