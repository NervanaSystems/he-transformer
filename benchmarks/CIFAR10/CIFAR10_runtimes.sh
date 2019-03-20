mkdir -p NEW

# Table row 1
NGRAPH_HE_SEAL_CONFIG=ckks_config_14_depth11.json \
NGRAPH_BATCH_DATA=1 \
NGRAPH_BATCH_TF=1 \
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_ENABLE_SERIALIZE=1 \
NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
python test.py --model=cnn \
--batch_size=4096 \
--batch_norm=True \
--train_poly_act=True \
--optimize_inference=False > NEW/cifar_L11_bnT_actT_optF.txt
mv tf_function_ngraph_cluster_0.json cifar_bnT_actT_optF.json

# Table row 2
NGRAPH_HE_SEAL_CONFIG=ckks_config_14_depth11.json \
NGRAPH_BATCH_DATA=1 \
NGRAPH_BATCH_TF=1 \
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_ENABLE_SERIALIZE=1 \
NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
python test.py --model=cnn \
--batch_size=4096 \
--batch_norm=True \
--train_poly_act=True \
--optimize_inference=True > NEW/cifar_L11_bnT_actT_optT.txt
mv tf_function_ngraph_cluster_0.json cifar_bnT_actT_optT.json

# Table row 3
NGRAPH_HE_SEAL_CONFIG=ckks_config_14_depth10.json \
NGRAPH_BATCH_DATA=1 \
NGRAPH_BATCH_TF=1 \
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_ENABLE_SERIALIZE=1 \
NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
python test.py --model=cnn \
--batch_size=4096 \
--batch_norm=True \
--train_poly_act=True \
--optimize_inference=True > NEW/cifar_L10_bnT_actT_optT.txt

# Table row 4
NGRAPH_HE_SEAL_CONFIG=ckks_config_14_depth11.json \
NGRAPH_BATCH_DATA=1 \
NGRAPH_BATCH_TF=1 \
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_ENABLE_SERIALIZE=1 \
NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
python test.py --model=cnn \
--batch_size=4096 \
--batch_norm=True \
--train_poly_act=False \
--optimize_inference=False > NEW/cifar_L11_bnT_actF_optF.txt
mv tf_function_ngraph_cluster_0.json cifar_bnT_actF_optF.json

# Table row 5
NGRAPH_HE_SEAL_CONFIG=ckks_config_14_depth11.json \
NGRAPH_BATCH_DATA=1 \
NGRAPH_BATCH_TF=1 \
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_ENABLE_SERIALIZE=1 \
NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
python test.py --model=cnn \
--batch_size=4096 \
--batch_norm=True \
--train_poly_act=False \
--optimize_inference=True > NEW/cifar_L11_bnT_actF_optT.txt
mv tf_function_ngraph_cluster_0.json cifar_bnT_actF_optT.json

# Table row 6
NGRAPH_HE_SEAL_CONFIG=ckks_config_14_depth10.json \
NGRAPH_BATCH_DATA=1 \
NGRAPH_BATCH_TF=1 \
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_ENABLE_SERIALIZE=1 \
NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
python test.py --model=cnn \
--batch_size=4096 \
--batch_norm=True \
--train_poly_act=False \
--optimize_inference=True > NEW/cifar_L10_bnT_actF_optT.txt

# Table row 7
NGRAPH_HE_SEAL_CONFIG=ckks_config_14_depth10.json \
NGRAPH_BATCH_DATA=1 \
NGRAPH_BATCH_TF=1 \
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_ENABLE_SERIALIZE=1 \
NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
python test.py --model=cnn \
--batch_size=4096 \
--batch_norm=False \
--train_poly_act=True \
--optimize_inference=False > NEW/cifar_L10_bnF_actT_optF.txt

# Table row 8
NGRAPH_HE_SEAL_CONFIG=ckks_config_14_depth10.json \
NGRAPH_BATCH_DATA=1 \
NGRAPH_BATCH_TF=1 \
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_ENABLE_SERIALIZE=1 \
NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
python test.py --model=cnn \
--batch_size=4096 \
--batch_norm=False \
--train_poly_act=False \
--optimize_inference=False > NEW/cifar_L10_bnF_actF_optF.txt


