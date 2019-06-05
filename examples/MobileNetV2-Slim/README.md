

# MobileNet V2 example

See here: https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
for a description. We use mobilenet_v2_0.35_96.

Make sure python env is active.

To get model and optimize for inference, call `./get_model.sh`

Also call `pip install pillow`

In one terminal, run
```
NGRAPH_ENABLE_CLIENT=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_BATCH_DATA=1 NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N13_L4.json NGRAPH_TF_BACKEND=HE_SEAL python test.py
```

In a second terminal on the same server, run
```
python client.py
```

##
For debugging purposes, run the model in plaintext with
```
NGRAPH_TF_BACKEND=HE_SEAL python test.py
```

# To download data
- First, create an account at imagenet.org
```bash
cd data
./download_and_convert_imagenet.sh ./
```

## Smaller parameters
For faster runtime (need to check accuracy), try
```
NGRAPH_ENABLE_CLIENT=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_BATCH_DATA=1 NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N12_L4_27bits.json NGRAPH_TF_BACKEND=HE_SEAL python test.py
```