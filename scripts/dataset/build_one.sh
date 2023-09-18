source config.sh
# Make output directories
echo $BUILD_PATH

# Run build
input_path_c=$INPUT_PATH/$c
build_path_c=$BUILD_PATH/$c
echo $input_path_c
echo $build_path_c

echo "Scaling meshes"
python $MESHFUSION_PATH/1_scale.py \
  --n_proc $NPROC \
  --in_dir $build_path_c/0_in \
  --out_dir $build_path_c/1_scaled \
  --t_dir $build_path_c/1_transform \
  --overwrite

echo "Create depths maps"
python $MESHFUSION_PATH/2_fusion.py \
  --mode=render --n_proc $NPROC \
  --in_dir $build_path_c/1_scaled \
  --out_dir $build_path_c/2_depth \
  --overwrite

echo "Produce watertight meshes"
python $MESHFUSION_PATH/2_fusion.py \
  --mode=fuse --n_proc $NPROC \
  --in_dir $build_path_c/2_depth \
  --out_dir $build_path_c/2_watertight \
  --t_dir $build_path_c/1_transform \
  --overwrite

#echo "Process watertight meshes"
#python ../sample_mesh.py \
#    --in_folder $build_path_c/2_watertight \
#    --n_proc $NPROC --resize \
#    --bbox_in_folder $build_path_c/0_in \
#    --pointcloud_folder $build_path_c/4_pointcloud \
#    --points_folder $build_path_c/4_points \
#    --voxels_folder $build_path_c/4_voxels \
#    --mesh_folder $build_path_c/4_watertight_scaled \
#    --packbits --float16

