#!/bin/bash

# Make sure the python environment is active
mkdir -p results

curr_dir=$(pwd)
echo $curr_dir

cd ../../examples/MobileNetV2

# Best performance
for i in {1..10}
do
  outfile=$curr_dir/results/035_224_clientless_${i}.txt
  echo "Trial ${i}"
  rm -rf $outfile
  touch $outfile
  start_batch=$((4096*i))
  echo "Start batch" $start_batch

  OMP_NUM_THREADS=56 \
  STOP_CONST_FOLD=1 \
  NGRAPH_VOPS=all \
  NGRAPH_COMPLEX_PACK=1 \
  NGRAPH_TF_BACKEND=HE_SEAL \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N12_L4.json \
  python test.py \
  --batch_size=4096 \
  --image_size=224 \
  --start_batch=$start_batch \
  --model=./model/mobilenet_v2_0.35_224_opt.pb \
  --data_dir=$DATA_DIR \
  --ngraph=true >> $outfile

done
cd -
