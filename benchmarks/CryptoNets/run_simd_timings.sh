#!/bin/bash

mkdir -p results/simd/
cd ../../build

for i in {1..10}
do
  echo "N=13 Trial $i"
  NGRAPH_ENCRYPT_DATA=1 \
    NGRAPH_BATCH_DATA=1 \
    NGRAPH_HE_SEAL_CONFIG=../test/model/he_seal_ckks_config_13.json \
    ./test/cryptonets_benchmark > ../benchmarks/CryptoNets/results/simd/exp13_${i}.txt

  echo "N=14 Trial $i"
  NGRAPH_ENCRYPT_DATA=1 \
    NGRAPH_BATCH_DATA=1 \
    NGRAPH_HE_SEAL_CONFIG=../test/model/he_seal_ckks_config_14.json \
    ./test/cryptonets_benchmark > ../benchmarks/CryptoNets/results/simd/exp14_${i}.txt
done
cd -
