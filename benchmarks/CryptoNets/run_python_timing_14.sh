# Make sure the python environment is active

cd ../../examples/cryptonets/
NGRAPH_TF_VLOG_LEVEL=5 NGRAPH_ENCRYPT_DATA=1 NGRAPH_BATCH_DATA=1 NGRAPH_BATCH_TF=1 NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_14.json NGRAPH_TF_BACKEND=HE:SEAL:CKKS python test.py
cd -