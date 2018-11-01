This example demonstrates a simple example of a small matrix multiplication and addition. This example depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/NervanaSystems/ngraph-tf). Make sure the python environment with the ngraph-tf bridge is active, i.e. run `source ~/repos/venvs/he3/bin/activate`.

```bash
NGRAPH_TF_BACKEND=HE:SEAL:CKKS python axpy.py
```
or
```bash
NGRAPH_TF_BACKEND=HE:SEAL:BFV python axpy.py
```
