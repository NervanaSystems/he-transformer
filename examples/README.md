This example demonstrates a simple example of a small matrix multiplication and addition. This example depends on the [ngraph-tensorflow bridge](https://github.com/NervanaSystems/ngraph-tensorflow-bridge/). Make sure the python environment with ng-tf bridge is active, i.e. run `source ~/repos/venvs/he3/bin/activate`.

```bash
NGRAPH_TF_BACKEND=HE:SEAL python axpy.py
```
or
```bash
NGRAPH_TF_BACKEND=HE:HEAAN python axpy.py
```
