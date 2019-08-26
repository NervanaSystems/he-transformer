#!/bin/bash

# Run it on skx159

# Make sure the python environment is active
mkdir -p results

# To prevent address in use error
lsof -i:34000 | tail -n1 | tr -s ' ' | cut -d ' '  -f 2 | xargs kill

curr_dir=$(pwd)
echo $curr_dir

cd ../../examples/MobileNetV2

mobilenet_dir=$(pwd)

# Best performance
rm -rf $curr_dir/results/035_96_ram_test.txt
touch $curr_dir/results/035_96_ram_test.txt

(NGRAPH_ENABLE_CLIENT=1 \
OMP_NUM_THREADS=56 \
STOP_CONST_FOLD=1 \
NGRAPH_VOPS=all \
NGRAPH_COMPLEX_PACK=1 \
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_TF_BACKEND=HE_SEAL \
NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N12_L4.json \
python test.py \
--batch_size=40  \
--image_size=96 \
--ngraph=true \
--model=./model/mobilenet_v2_0.35_96_opt.pb \
--data_dir=$DATA_DIR \
--ngraph=true) & >> $curr_dir/results/035_96_ram_test.txt

sleep 15s
echo "Starting ssh command"

ssh 10.52.122.144 'cd /nfs/pdx/home/fboemer/repos/he-transformer/results/mobilenet_results && bash -s ' < /nfs/pdx/home/fboemer/repos/he-transformer/results/mobilenet_results/run_035_96_ram_test_client.sh

echo "Done with ssh command"

cd -