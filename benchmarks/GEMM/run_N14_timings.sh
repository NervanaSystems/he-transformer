OMP_NUM_THREADS=1 \
  NGRAPH_HE_SEAL_CONFIG=he_seal_ckks_config_14.json \
  NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_OPTIMIZED_MULT=1 \
  python gemm_timings.py --out=./gemm_14_results.txt
