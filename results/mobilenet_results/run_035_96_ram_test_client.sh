#!/bin/bash

mkdir -p results

# To prevent address in use error
# lsof -i:34000 | tail -n1 | tr -s ' ' | cut -d ' '  -f 2 | xargs kill

curr_dir=$(pwd)
echo $curr_dir

cd ../../examples/MobileNetV2

echo "Runnign ssh command"

source /nfs/pdx/home/fboemer/repos/he-transformer/build/external/venv-tf-py3/bin/activate;
echo "Sourced env"
cd /nfs/pdx/home/fboemer/repos/he-transformer/examples/MobileNetV2/;
OMP_NUM_THREADS=56 \
NGRAPH_COMPLEX_PACK=1 \
python client.py \
--batch_size=40 \
--image_size=96 \
--data_dir=/nfs/pdx/home/fboemer/repos/data/ImageNet \
--hostname=10.52.122.146 >> /nfs/pdx/home/fboemer/repos/he-transformer/results/mobilenet_results/035_96_ram_test.txt;

echo "Done with client command"

cd -


# To get PID with highest memory


# top=$(ps aux --sort=-%mem | awk 'NR<=2{print $0}' | tail -n 1 | tr -s ' ' | cut -d ' ' -f2); echo $top; psrecord $top --log server_ram_96.txt --plot server_ram_96.png --interval 1