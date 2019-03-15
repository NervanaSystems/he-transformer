This directory provides a framework for developing new HE models and exporting them with ngraph-tf.

* `train.py` is used to train the models. Currently, we have a simple CNN model with two parameters: `batch_norm` and `train_poly_act`
  * if `batch_norm` is enabled (i.e. `batch_norm=True`), a batch norm layer will follow each convolution layer
  * if `train_poly_act` is enabled, the polynomial activations will be of the form `ax^2 + bx`, where `a=0, b=1`, intitially, but `a,b` are trained. If `train_poly_act` is not enabled, the polynomial activation is fixed at `0.125x^2 + 0.5x + 0.25`
* `test.py` is used to perform inference.
HE-transformer expects serialized models to have inputs as placeholders, and model weights stored as constants.
* `test.py` also performs this serialization. Finally, `test.py` also optimizes the model for inference, for example by folding batch norm weights into convolution weights to reduce the mutliplicative depth of the model.

# 1 Train a model
```python
python train.py --model=cnn --batch_norm=True --train_poly_act=True [--max_steps=10000]
```

## To resume training a model
```python
python train.py --model=cnn --resume=True
```

# 2. Run trained model:
## Skip inference, just export serialized graph:
```python
NGRAPH_TF_BACKEND=NOP NGRAPH_ENABLE_SERIALIZE=1 python test.py --model=cnn --batch_norm=True --train_poly_act=True --batch_size=1
```

## Run inference on CPU backend:
```python
python test.py --model=cnn --batch_norm=True --train_poly_act=True --batch_size=10000
```

## Run inference on HE backend
```python
NGRAPH_HE_SEAL_CONFIG=ckks_config_13_depth12.json NGRAPH_BATCH_DATA=1 NGRAPH_BATCH_TF=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_TF_BACKEND=HE_SEAL_CKKS python test.py --model=cnn --batch_size=1000
```

# 3. To run a complete set of timing results, with batch-norm folding
```sh
# Run 10 rounds of training and parse results
./training_accuracies.sh
./parse_training_accuracies_results.sh
# Run 10 rounds of inference and parse results
./inference_runtimes.sh
python parse_inference_runtimes.py
```

# 4. To run a complete set of timing results, without batch-norm folding
```sh
# Run 10 rounds of training
./training_accuracies.sh
# Run 10 rounds of inference and parse results
./inference_nobn_runtimes.sh
```
Then run the jupyter notebook  `no_bn_results.ipynb`