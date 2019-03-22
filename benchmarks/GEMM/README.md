This folder is used to generate runtime results for the GEMM example (Figure 5)

This example demonstrates a simple example of a small matrix multiplication and addition. This example depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/NervanaSystems/ngraph-tf). Make sure the python environment with the ngraph-tf bridge is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`.

Call `./run_N13_timings.sh` to generate `gemm_13_results.txt`
Call `./run_N14_timings.sh` to generate `gemm_14_results.txt`