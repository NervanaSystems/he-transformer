#!/bin/bash

# Make sure the python environment is active
mkdir -p results

curr_dir=$(pwd)
echo $curr_dir

cd ../../examples/MobileNetV2

# Best performance
for i in {1..1}
do
  outfile=$curr_dir/results/best_${i}.txt
  echo "Trial ${i}"
  rm -rf $outfile
  touch $outfile

  (OMP_NUM_THREADS=56 \
  NGRAPH_ENABLE_CLIENT=1 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_TF_BACKEND=HE_SEAL \
  python test.py \
  --batch_size=3  \
  --data_dir=$DATA_DIR &) >> $outfile

  # Let server start
  sleep 5s
  OMP_NUM_THREADS=56 \
  python client.py \
  --batch_size=3 \
  --data_dir=$DATA_DIR >> $outfile


done

cd -