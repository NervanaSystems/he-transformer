

mkdir -p runtimes


# BNN-BN row 1 ~54s * 11 trials
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_BATCH_DATA=1 \
NGRAPH_BATCH_TF=1 \
NGRAPH_HE_SEAL_CONFIG=./cryptonets_bnn_N16384_q9x30.json \
NGRAPH_TF_BACKEND=HE_SEAL_CKKS python test.py \
--batch_size=4096 > runtimes/BNN_BN_row1.txt

# BNN-BN row 2 ~52s * 11 trials
NGRAPH_OPTIMIZED_MULT=1 \
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_BATCH_DATA=1 \
NGRAPH_BATCH_TF=1 \
NGRAPH_HE_SEAL_CONFIG=./cryptonets_bnn_N16384_q9x30.json \
NGRAPH_TF_BACKEND=HE_SEAL_CKKS python test.py \
--batch_size=4096 > runtimes/BNN_BN_row2.txt

# BNN-BN row 3 ~17s * 11 trials
NGRAPH_OPTIMIZED_MULT=1 \
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_BATCH_DATA=1 \
NGRAPH_BATCH_TF=1 \
NGRAPH_HE_SEAL_CONFIG=./cryptonets_bnn_N8192_q7x30.json \
NGRAPH_TF_BACKEND=HE_SEAL_CKKS python test.py \
--batch_size=4096 > runtimes/BNN_BN_row3.txt