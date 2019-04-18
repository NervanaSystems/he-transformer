

Run


`pip install pillow`

`./get_model.sh`

In one terminal,
```
NGRAPH_ENABLE_CLIENT=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_BATCH_DATA=1 NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N13_L3.json NGRAPH_TF_BACKEND=HE_SEAL_CKKS python test.py --batch_size=1
```

In another terminal on same server,
```
python client.py
```