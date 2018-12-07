This directory provides a framework for developing new HE models and exporting them with ngraph-tf.

# 1 Train a model
```python
python train.py --model='CryptoDL'
```

# 2 Save the model weights
```python
python save_weights.py --model='CryptoDL'
```

# 3 Export model with ng-tf
```python
NGRAPH_TF_BACKEND=NOP NGRAPH_ENABLE_SERIALIZE=1 python export_model.py --model='CryptoDL'
```

This will export the model as `tf_function_ngraph_cluster_0.json` or similar
