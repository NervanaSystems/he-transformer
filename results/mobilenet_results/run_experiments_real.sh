#!/bin/bash

# Make sure the python environment is active
mkdir -p results

curr_dir=$(pwd)
echo $curr_dir

cd ../../examples/MobileNetV2

# Best performance
for i in {1..10}
do
  outfile=$curr_dir/results/best_${i}.txt
  echo "Trial ${i}"
  rm -rf $outfile
  touch $outfile

  (OMP_NUM_THREADS=56 \
  NGRAPH_ENABLE_CLIENT=1 \
  STOP_CONST_FOLD=1 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_TF_BACKEND=HE_SEAL \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N12_L4.json \
  python test.py \
  --batch_size=2048  \
  --data_dir=$DATA_DIR \
  --ngraph=true &) >> $outfile

  # Let server start
  sleep 5s
  OMP_NUM_THREADS=56 \
  python client.py \
  --batch_size=2048 \
  --data_dir=$DATA_DIR >> $outfile

done
cd -