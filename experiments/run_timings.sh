OMP_NUM_THREADS=1 NGRAPH_HE_SEAL_CONFIG=he_seal_ckks_config_13.json NGRAPH_TF_BACKEND=HE:SEAL:CKKS python gemm_timings.py --out=./gemm_13_results.txt
OMP_NUM_THREADS=1 NGRAPH_HE_SEAL_CONFIG=he_seal_ckks_config_14.json NGRAPH_TF_BACKEND=HE:SEAL:CKKS python gemm_timings.py --out=./gemm_14_results.txt

