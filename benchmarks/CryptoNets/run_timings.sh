#!/bin/bash

# Make sure the python environment is active
mkdir -p results
cd ../../examples/MNIST-Cryptonets/

# For some reason, normal piping of stdout doesn't capture the tensorflow output,
# hence the ugly lines below.

echo 'N=13, bs=1'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_13.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=1' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp13_bs1.txt

echo 'N=13, bs=2'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_13.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=2' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp13_bs2.txt

echo 'N=13, bs=4'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_13.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=4' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp13_bs4.txt

echo 'N=13, bs=8'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_13.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=8' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp13_bs8.txt

echo 'N=13, bs=16'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_13.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=16' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp13_bs16.txt

echo 'N=13, bs=32'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_13.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=32' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp13_bs32.txt

echo 'N=13, bs=64'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_13.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=64' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp13_bs64.txt

echo 'N=13, bs=128'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_13.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=128' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp13_bs128.txt

echo 'N=13, bs=256'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_13.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=256' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp13_bs256.txt

echo 'N=13, bs=512'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_13.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=512' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp13_bs512.txt

echo 'N=13, bs=1024'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_13.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=1024' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp13_bs1024.txt

echo 'N=13, bs=2048'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_13.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=2048' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp13_bs2048.txt

echo 'N=13, bs=4096'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_13.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=4096' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp13_bs4096.txt


echo 'N=14, bs=1'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_14.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=1' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp14_bs1.txt

echo 'N=14, bs=2'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_14.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=2' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp14_bs2.txt

echo 'N=14, bs=4'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_14.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=4' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp14_bs4.txt

echo 'N=14, bs=8'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_14.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=8' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp14_bs8.txt

echo 'N=14, bs=16'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_14.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=16' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp14_bs16.txt

echo 'N=14, bs=32'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_14.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=32' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp14_bs32.txt

echo 'N=14, bs=64'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_14.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=64' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp14_bs64.txt

echo 'N=14, bs=128'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_14.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=128' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp14_bs128.txt

echo 'N=14, bs=256'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_14.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=256' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp14_bs256.txt

echo 'N=14, bs=512'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_14.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=512' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp14_bs512.txt

echo 'N=14, bs=1024'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_14.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=1024' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp14_bs1024.txt

echo 'N=14, bs=2048'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_14.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=2048' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp14_bs2048.txt

echo 'N=14, bs=4096'
sh -c 'NGRAPH_TF_VLOG_LEVEL=5 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_BATCH_DATA=1 \
  NGRAPH_BATCH_TF=1 \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_14.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  python test.py --batch_size=4096' 2>&1 | tee ../../benchmarks/CryptoNets/results/exp14_bs4096.txt

cd -