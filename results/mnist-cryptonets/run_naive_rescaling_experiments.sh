#!/bin/bash

# Make sure the python environment is active
mkdir -p results
cd ../../examples/MNIST-Cryptonets/

echo $pwd

outfolder=../../results/mnist-cryptonets/results

# naive rescaling
for i in {1..10}
do
  outfile=$outfolder/naive_recsaling_${i}.txt
  OMP_NUM_THREADS=1 \
  NGRAPH_ENCRYPT_DATA=1 \
  NAIVE_RESCALING=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N13_L7.json \
  NGRAPH_TF_BACKEND=HE_SEAL \
  python test.py --batch_size=4096 > $outfile
done

cd -