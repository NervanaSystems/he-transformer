This example demonstrates a simple example of a small matrix multiplication and addition. This example depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/NervanaSystems/ngraph-tf). Make sure the python environment with the ngraph-tf bridge is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`.

To run on the CKKS backend,
```bash
NGRAPH_TF_BACKEND=HE_SEAL_CKKS python axpy.py
```
To run on the BFV backend,
```bash
NGRAPH_TF_BACKEND=HE_SEAL_BFV python axpy.py
```

Note, the BFV encryption scheme suports only integers. For floating-point support, use the CKKS encryption scheme.