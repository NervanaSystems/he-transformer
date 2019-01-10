This directory provides a framework for developing new HE models and exporting them with ngraph-tf.

`train.py` is used to train the models.
`test.py` is used to perform inference.
HE-transformer expects serialized models to have inputs as placeholders, and model weights stored as constants.
`test.py` also performs this serialization. Finally, `test.py` also optimizes the model for inference, for example by folding batch norm weights into convolution weights to reduce the mutliplicative depth of the model.

# 1 Train a model
```python
python train.py --model=CNN
```

## To resume training a model
```python
python train.py --model=CNN --resume=True
```

# 2. Run trained model:
## Skip inference, just export serialized graph:
```python
NGRAPH_TF_BACKEND=NOP NGRAPH_ENABLE_SERIALIZE=1 python test.py --model=CNN --batch_size=1 --report_accuracy=0
```

## Run inference on CPU backend:
```python
python test.py --model=CNN --batch_size=100
```

## Run inference on HE backend
```python
NGRAPH_HE_SEAL_CONFIG=ckks_config_13_depth12.json NGRAPH_BATCH_DATA=1 NGRAPH_BATCH_TF=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_TF_BACKEND=HE_SEAL_CKKS python test.py --model=cnn --batch_size=1000
#  0.571% accuracy, 1072824ms; on CPU, 0.629% accuracy.
# NGRAPH_BATCH_DATA=1 NGRAPH_BATCH_TF=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_TF_BACKEND=HE_SEAL_CKKS python test.py --model=CNN --batch_size=100
```