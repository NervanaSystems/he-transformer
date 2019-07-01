#!/bin/bash

# Make sure the python environment is active
mkdir -p results
cd ../../examples/MNIST-Cryptonets/

echo $pwd

outfolder=../../results/mnist-cryptonets/results

# tune OMP_NUM_THREADS
for nt in {56,32,16,8,4,2,1} #1, 2, 4, 8, 16, 32, 56}
  do
  for i in {1..10}
  do
    outfile=$outfolder/best_nt${nt}_${i}.txt
    OMP_NUM_THREADS=${nt} \
    NGRAPH_ENCRYPT_DATA=1 \
    NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N13_L7.json \
    NGRAPH_TF_BACKEND=HE_SEAL \
    python test.py --batch_size=4096 > $outfile
  done
done
