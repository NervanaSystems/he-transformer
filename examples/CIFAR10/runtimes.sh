
# CNN True True
# 792s for batch of 1000, accuracy 61.5
NGRAPH_HE_SEAL_CONFIG=ckks_config_13_depth12.json NGRAPH_BATCH_DATA=1 NGRAPH_BATCH_TF=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_TF_BACKEND=HE_SEAL_CKKS python test.py --model=cnn --batch_size=4096 --batch_norm=True --train_poly_act=True


# CNN True True (batch size 4096) - 62.4% accuracy; 867049ms => 200ms / image
NGRAPH_HE_SEAL_CONFIG=ckks_config_13_depth12.json NGRAPH_BATCH_DATA=1 NGRAPH_BATCH_TF=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_TF_BACKEND=HE_SEAL_CKKS python test.py --model=cnn --batch_size=4096 --batch_norm=True --train_poly_act=True



# DeeperCNN True, True (batch 4096) - 60.93% accuracy; 1034467ms => 250ms / image
NGRAPH_HE_SEAL_CONFIG=ckks_config_13_depth12.json NGRAPH_BATCH_DATA=1 NGRAPH_BATCH_TF=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_TF_BACKEND=HE_SEAL_CKKS python test.py --model=deeper_cnn --batch_size=4096 --batch_norm=True --train_poly_act=True
