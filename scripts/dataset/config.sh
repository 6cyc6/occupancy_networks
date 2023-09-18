ROOT=..

export MESHFUSION_PATH=../../external/mesh-fusion
export HDF5_USE_FILE_LOCKING=FALSE # Workaround for NFS mounts

INPUT_PATH=$ROOT/data/external
BUILD_PATH=$ROOT/data/build
OUTPUT_PATH=$ROOT/data/obj

NPROC=12
TIMEOUT=180
N_VAL=100
N_TEST=100
N_AUG=50

c=cup

# Utility functions
lsfilter() {
 folder=$1
 other_folder=$2
 ext=$3

 for f in $folder/*; do
   filename=$(basename $f)
   if [ ! -f $other_folder/$filename$ext ] && [ ! -d $other_folder/$filename$ext ]; then
    echo $filename
   fi
 done
}
