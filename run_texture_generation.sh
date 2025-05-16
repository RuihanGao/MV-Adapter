export LD_LIBRARY_PATH=/data/ruihan/anaconda3/envs/midi/lib/python3.10/site-packages/pymeshlab/lib/:$LD_LIBRARY_PATH # to avoid error "undefined symbol: _ZdlPvm, version Qt_5"

export midi_dir="/data/ruihan/projects/MIDI-3D"

# 2025-05-15. Test on 13 prompts

# Example command to generate textures for a single image and mesh
CUDA_VISIBLE_DEVICES=0 python -m scripts.texture_i2tex --image $midi_dir/assets/data/midi_images_20250515/08_fruit_basket.png --mesh $midi_dir/output/08_fruit_basket.glb --save_dir output --save_name 08_fruit_basket --remove_bg

# # Create a for loop to run the above command for all images in the directory $midi_dir/assets/data/midi_images_20250515
# for image in $midi_dir/assets/data/midi_images_20250515/*.png; do
#     # get the base name of the image file
#     base_name=$(basename "$image" .png)
#     echo "Processing image: $base_name"
#     # run the command with the image and mesh file
#     CUDA_VISIBLE_DEVICES=0 python -m scripts.texture_i2tex --image "$image" --mesh "$midi_dir/output/${base_name}.glb" --save_dir output --save_name "$base_name" --remove_bg
# done
