#!/bin/bash

for i in {1..10}
do
  echo "Trial $i"
  OMP_NUM_THREADS=1 NGRAPH_HE_SEAL_CONFIG=./bn_config.json ../../build/test/unit-test --gtest_filter="HE_SEAL_CKKS.batch_norm_fusion_he_large" > exp_${i}.txt
done
