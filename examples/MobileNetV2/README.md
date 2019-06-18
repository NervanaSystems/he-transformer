# MobileNet V2 example

This folder demonstrates an example of inference on MobileNetV2.
Note: this is a work in progress, and requires ~150GB memory.
Runtime will be very slow without many cores.

See here: https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
for a description. We use mobilenet_v2_0.35_96 and resize images to `84x84x3`

# To peform inference
1. Make sure python env is active, i.e. run
```bash
cd $HE_TRANSFORMER/build
source external/venv-tf-py3/bin/activat
```
Also be sure the `he_seal_client` wheel has been installed

2. Add TensorFlow graph transforms to your path, i.e. run
```bash
export PATH=$HE_TRANSFORMER/build/ext_ngraph_tf/src/ext_ngraph_tf/build_cmake/tensorflow/bazel-bin/tensorflow/tools/graph_transforms:$PATH
```

2. To download the model and optimize for inference, call
```bash
./get_model.sh
```

3. To enable image processing, run
```bash
pip install pillow
```

4. To perform inference
    1. In one terminal, run
      ```bash
      NGRAPH_ENABLE_CLIENT=1 \
      NGRAPH_ENCRYPT_DATA=1 \
      NGRAPH_BATCH_DATA=1 \
      NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N12_L4.json \
      NGRAPH_TF_BACKEND=HE_SEAL \
      python test.py
      ```

    2. In a second terminal on the same server (with the python env active), run
    ```bash
    python client.py
    ```
Upon successful completion, the client will output the top5 categories for an image of Grace Hopper in a military uniform:
```
top5 [922 458 721 653 835]
['book jacket' 'bow tie' 'pill bottle' 'military uniform' 'suit']
```

# Debugging
1. For debugging purposes, run the model in plaintext with
```bash
NGRAPH_TF_BACKEND=HE_SEAL python test.py
```

2. To run the model without the client, (encryption and decryption will occur locally, so this isn't privacy-preserving):
```bash
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_BATCH_DATA=1 \
NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N12_L4.json \
NGRAPH_TF_BACKEND=HE_SEAL \
python test.py
```

# Fastest
For faster runtime, try
```bash
OMP_NUM_THREADS=56 \
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_BATCH_DATA=1 \
NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N12_L4.json \
NGRAPH_TF_BACKEND=HE_SEAL \
python test.py
```

# Image-Net evaluation


1. First, sign up for an account at image-net.org
2. Download the 2012 test_images (all tasks)) 13GB MD5: fe64ceb247e473635708aed23ab6d839

3. ```bash
tar -xf ILSVRC2012_img_test.tar
```

4. To crop images to 84x84
```bash
for i in $(ls *.JPEG); do convert -define jpeg:size=84x84 $i -thumbnail 84x84^ -gravity center -extent 84x84x "${i}_crop.jpeg"; done
```

3. Download development kit (Task 1 & 2) and extract validation_ground_truth.txt
