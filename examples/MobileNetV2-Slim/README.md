

# MobileNet V2 example

Make sure python env is active.

To get model and optimize for inference, call `./get_model.sh`

Also call `pip install pillow`

In one terminal, run
```
NGRAPH_ENABLE_CLIENT=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_BATCH_DATA=1 NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N13_L4.json NGRAPH_TF_BACKEND=HE_SEAL python test.py --batch_size=1
```


In a second terminal on the same server, run
```
python client.py
```