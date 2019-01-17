#!/bin/bash

mkdir -p results
cd ../../examples/MNIST-Cryptonets/

# For some reason, normal piping of stdout doesn't capture the tensorflow output,

echo 'N=13, bs=1'
# Encrypt data
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_13.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=1' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp13_bs1_enc_data.txt

# Encrypt model
  sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_MODEL=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_OPTIMIZED_MULT=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_13.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=1' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp13_bs1_enc_model.txt

# Encrypt data and model
  sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_MODEL=1 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_13.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=1' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp13_bs1_enc_both.txt

echo 'N=14, bs=1'
# Encrypt data
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_14.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=1' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp14_bs1_enc_data.txt

# Encrypt model
  sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_MODEL=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_OPTIMIZED_MULT=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_14.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=1' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp14_bs1_enc_model.txt

# Encrypt data and model
  sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_MODEL=1 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_14.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=1' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp14_bs1_enc_both.txt

cd -