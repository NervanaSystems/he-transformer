#!/bin/bash

# Make sure the python environment is active
mkdir -p results

curr_dir=$(pwd)
echo $curr_dir

cd ../../examples/MobileNetV2

# 0.35 results

# top1_acc 42.37
# top5_acc 67.106
python test.py \
  --batch_size=50000  \
  --image_size=96 \
  --model=./model/mobilenet_v2_0.35_96_opt.pb \
  --data_dir=$DATA_DIR > $curr_dir/results/acc_035_96.txt

#  top1_acc 50.032
#  top5_acc 74.382
python test.py \
  --batch_size=50000  \
  --image_size=128 \
  --model=./model/mobilenet_v2_0.35_128_opt.pb \
  --data_dir=$DATA_DIR > $curr_dir/results/acc_035_128.txt

# top1_acc 56.202
# top5_acc 79.73
python test.py \
  --batch_size=50000  \
  --image_size=160 \
  --model=./model/mobilenet_v2_0.35_160_opt.pb \
  --data_dir=$DATA_DIR > $curr_dir/results/acc_035_160.txt

python test.py \
  --batch_size=50000  \
  --image_size=192 \
  --model=./model/mobilenet_v2_0.35_192_opt.pb \
  --data_dir=$DATA_DIR > $curr_dir/results/acc_035_192.txt

python test.py \
  --batch_size=50000  \
  --image_size=224 \
  --model=./model/mobilenet_v2_0.35_224_opt.pb \
  --data_dir=$DATA_DIR > $curr_dir/results/acc_035_224.txt

# 0.50 results
python test.py \
  --batch_size=50000  \
  --image_size=96 \
  --model=./model/mobilenet_v2_0.5_96_opt.pb \
  --data_dir=$DATA_DIR > $curr_dir/results/acc_050_96.txt

python test.py \
  --batch_size=50000  \
  --image_size=128 \
  --model=./model/mobilenet_v2_0.5_128_opt.pb \
  --data_dir=$DATA_DIR > $curr_dir/results/acc_050_128.txt

python test.py \
  --batch_size=50000  \
  --image_size=160 \
  --model=./model/mobilenet_v2_0.5_160_opt.pb \
  --data_dir=$DATA_DIR > $curr_dir/results/acc_050_160.txt

python test.py \
  --batch_size=50000  \
  --image_size=192 \
  --model=./model/mobilenet_v2_0.5_192_opt.pb \
  --data_dir=$DATA_DIR > $curr_dir/results/acc_050_192.txt

python test.py \
  --batch_size=50000  \
  --image_size=224 \
  --model=./model/mobilenet_v2_0.5_224_opt.pb \
  --data_dir=$DATA_DIR > $curr_dir/results/acc_050_224.txt

# 0.75 results
python test.py \
  --batch_size=50000  \
  --image_size=96 \
  --model=./model/mobilenet_v2_0.75_96_opt.pb \
  --data_dir=$DATA_DIR > $curr_dir/results/acc_075_96.txt

python test.py \
  --batch_size=50000  \
  --image_size=128 \
  --model=./model/mobilenet_v2_0.75_128_opt.pb \
  --data_dir=$DATA_DIR > $curr_dir/results/acc_075_128.txt

python test.py \
  --batch_size=50000  \
  --image_size=160 \
  --model=./model/mobilenet_v2_0.75_160_opt.pb \
  --data_dir=$DATA_DIR > $curr_dir/results/acc_075_160.txt

python test.py \
  --batch_size=50000  \
  --image_size=192 \
  --model=./model/mobilenet_v2_0.75_192_opt.pb \
  --data_dir=$DATA_DIR > $curr_dir/results/acc_075_192.txt

# 0.75 / 224 runs out of memory
#python test.py \
#  --batch_size=50000  \
#  --image_size=224 \
#  --model=./model/mobilenet_v2_0.75_224_opt.pb \
#  --data_dir=$DATA_DIR > $curr_dir/results/acc_075_224.txt

cd -