#!/bin/bash

mkdir -p results
for i in {1..10}
do
  echo "8192 Trial $i"
  OMP_NUM_THREADS=1 NGRAPH_HE_SEAL_CONFIG=./bn_config_8192.json ../../build/test/unit-test --gtest_filter="HE_SEAL_CKKS.batch_norm_fusion_he_large" > results/exp8192_${i}.txt
done

for i in {1..10}
do
  echo "16384 Trial $i"
  OMP_NUM_THREADS=1 NGRAPH_HE_SEAL_CONFIG=./bn_config_16384.json ../../build/test/unit-test --gtest_filter="HE_SEAL_CKKS.batch_norm_fusion_he_large" > results/exp16384_${i}.txt
done