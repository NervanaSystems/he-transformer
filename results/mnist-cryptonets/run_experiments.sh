#!/bin/bash

# Make sure the python environment is active
mkdir -p results
cd ../../examples/MNIST-Cryptonets/

echo $pwd


for i in {1..10}
do
  # Best performance
  OMP_NUM_THREADS=56 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N13_L7.json \
  NGRAPH_TF_BACKEND=HE_SEAL \
  python test.py --batch_size=4096 > ../../results/mnist-cryptonets/results/best_${i}.txt

  # 1 thread
  OMP_NUM_THREADS=1 NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N13_L7.json \
  NGRAPH_TF_BACKEND=HE_SEAL \
  python test.py --batch_size=4096 > ../../results/mnist-cryptonets/results/OMP1_${i}.txt

  # naive rescaling
  OMP_NUM_THREADS=1 \
  NGRAPH_ENCRYPT_DATA=1 \
  NAIVE_RESCALING=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N13_L7.json \
  NGRAPH_TF_BACKEND=HE_SEAL \
  python test.py --batch_size=4096 > ../../results/mnist-cryptonets/results/naive_recsaling_${i}.txt

done
cd -