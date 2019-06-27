

To test naive rescaling:
```
NAIVE_RESCALING=1 \
OMP_NUM_THREADS=1 \
NGRAPH_VOPS=all \
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N13_L7.json \
NGRAPH_TF_BACKEND=HE_SEAL \
python test.py --batch_size=4096
```
Takes about 65s

To test lazy rescaling:
```
OMP_NUM_THREADS=1 \
NGRAPH_VOPS=all \
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N13_L7.json \
NGRAPH_TF_BACKEND=HE_SEAL \
python test.py --batch_size=4096
```
Takes about 8.210s