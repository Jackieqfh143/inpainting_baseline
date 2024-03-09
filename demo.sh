export TORCH_HOME=$(pwd) &&
export PYTHONPATH="$(pwd)/src/lib/FcF:$(pwd)/src/lib/MAT:"
CUDA_VISIBLE_DEVICES=0
python demo.py \
--device "cpu" \
--model_name  CoModGAN \
--model_path ./checkpoints/CoModGAN/places/comodgan_256_places2.pt \
--target_size 256 \
--sample_num 1 \
--img_path ./example/place/imgs \
--mask_path ./example/place/masks \
--save_path ./demo_results_mat \
--max_num 100
