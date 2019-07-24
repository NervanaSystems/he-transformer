#!/bin/bash

# Make sure the python environment is active
mkdir -p results

curr_dir=$(pwd)
echo $curr_dir

cd ../../examples/MNIST-Cryptonets-Relu

# Best performance
for i in {1..10}
do
  outfile=$curr_dir/results/wan_${i}.txt
  echo "Trial ${i}"
  rm -rf $outfile
  touch $outfile

  (NGRAPH_ENABLE_CLIENT=1 \
  OMP_NUM_THREADS=24 \
  NGRAPH_VOPS=total \
  NGRAPH_COMPLEX_PACK=1 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_TF_BACKEND=HE_SEAL \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N11_L1.json \
  python test.py \
  --batch_size=2048 &) >> $outfile

  # Wait for server to start
  sleep 3s

  ssh skx113 << EOF
    export HE_TRANSFORMER=/nfs/site/home/fboemer/repos/he-transformer/;
    source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate;
    outfile=$HE_TRANSFORMER/results/mnist-cryptonets-relu/results/cryptonets_relu_client_wan.txt;
    cd $HE_TRANSFORMER/examples/MNIST-Cryptonets-Relu/;
    OMP_NUM_THREADS=24 \
    NGRAPH_COMPLEX_PACK=1 \
    python ../pyclient_mnist.py \
    --batch_size=2048 \
    --hostname=skx114 >> $outfile;
EOF

  # Wait to finish
  sleep 3s

done
cd -