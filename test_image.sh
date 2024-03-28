# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1
cur_path=$(pwd)
file_path=$(dirname $0)
python3 -u ${cur_path}/${file_path}/image2pyramid4_test.py \
 2>&1 | tee ${cur_path}/${file_path}/temp_test.log