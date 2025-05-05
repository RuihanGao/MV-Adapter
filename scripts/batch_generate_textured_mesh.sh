# Iterate a list of experiments. 
# For each experiment, run the inference script to generate multi-view images and textured meshes.
experiments=("stacked_cups" "fruit_basket" "headphone_stand" "vase_table")
gpu_id=0 
for experiment in "${experiments[@]}"; do
    echo "Running experiment: $experiment"
    CUDA_VISIBLE_DEVICES=$gpu_id python scripts/inference_ig2mv_sdxl.py --mesh data/${experiment}_unwrapped.obj --image data/${experiment}.png --output output/${experiment}_output.png --save_3D
done