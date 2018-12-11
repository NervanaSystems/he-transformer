This directory provides a framework for developing new HE models and exporting them with ngraph-tf.

# 1 Train a model
```python
python train.py --model=simple
```

# Export model with ng-tf
```
NGRAPH_TF_BACKEND=NOP NGRAPH_ENABLE_SERIALIZE=1 python optimize_for_inference.py --model=simple
```

This will export the model as `tf_function_ngraph_cluster_0.json` or similar.

Currently, the BatchNorm node is a BatchNormTraining node, followed by a GetOutputElement.
This is because the saved model is based on the training.
Instead, I think we need to pass self.training as a boolean placeholder.

For now, as a workaround, we should try:
1) Register "GetOutputElementElimination" pass to eliminate the GetOutputElement node.
2) Implement "BatchNormTraining" op in he_backend.

Once BatchNorm node is used instead, implementing BatchNormInference op should be nearly identitcal to the BatchNorm op.
