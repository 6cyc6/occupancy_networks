source config.sh
# Make output directories

# Run build
input_path_c=$INPUT_PATH/$c
build_path_c=$BUILD_PATH/$c
echo $input_path_c
echo $build_path_c

mkdir -p $build_path_c/5_obj \

echo "Converting meshes to OFF"
lsfilter $input_path_c $build_path_c/5_obj .off | parallel -P $NPROC --timeout $TIMEOUT\
   meshlabserver -i $build_path_c/4_watertight_scaled/{}.off -o $build_path_c/5_obj/{}.obj;
