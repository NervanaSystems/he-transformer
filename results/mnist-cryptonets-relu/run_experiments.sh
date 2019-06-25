#!/bin/bash

# Make sure the python environment is active
mkdir -p results

curr_dir=$(pwd)
echo $curr_dir

cd ../../examples/MNIST-Cryptonets-Relu/

for i in {1..10}
do
  # Best
  outfile=$curr_dir/results/best_${i}.txt
  echo "Trial ${i}"
  rm -rf $outfile
  touch $outfile

  (OMP_NUM_THREADS=56 \
  NGRAPH_ENABLE_CLIENT=1 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N11_L1.json \
  NGRAPH_TF_BACKEND=HE_SEAL \
  python test.py --batch_size=1024 &) >> $outfile

  # Let server start
  sleep 3s
  OMP_NUM_THREADS=56 python ../pyclient_mnist.py --batch_size=1024 >> $outfile

  # Wait to finish
  sleep 3s
done

cd -