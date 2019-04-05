This example demonstrates the [Cryptonets](https://www.microsoft.com/en-us/research/publication/cryptonets-applying-neural-networks-to-encrypted-data-with-high-throughput-and-accuracy/) network, which achieves ~99% accuracy on MNIST.

Note: performance is greatly improved by use of parallelism. Make sure OpenMP is installed and utilizing available cores. If you run out of memory, or the test takes too long, you are better off only running the C++ unit-tests `./test/unit-test` from the build folder instead.

With `OMP_NUM_THREADS=4` and using the smaller of two parameter settings, `he_seal_config_N13_L7.json`, the model requires ~45GB of memory.

This example depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/NervanaSystems/ngraph-tf). Make sure the python environment with ngraph-tf bridge is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`.

# Train the network
First, train the network using
```
python train.py
```
This trains the network briefly and stores the network weights.

# Test the network
## Python
To test the network, run
```
[NGRAPH_ENCRYPT_DATA=1] [NGRAPH_ENCRYPT_MODEL=1] NGRAPH_BATCH_DATA=1 NGRAPH_BATCH_TF=1 NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N13_L7.json NGRAPH_TF_BACKEND=HE_SEAL_CKKS python test.py --batch_size=4096 --report_accuracy=1
```
This runs inference on the Cryptonets network using the SEAL CKKS backend.
The `he_seal_ckks_config_N13_L7.json` file specifies the parameters which to run the model on. You can also use the `he_seal_ckks_config_N14_L7.json` or create your own configuration. Note: the batch size must be beweteen 1 and 4096.

To export the serialized model for use in C++ integration with nGraph, run
```
NGRAPH_ENABLE_SERIALIZE=1 python test.py --save_batch=1 [--batch_size=BATCH_SIZE]
```

This will generate:
* `mnist_cryptonets_batch_[BATCH_SIZE].json`, which is the serialized nGraph computation graph.
* `x_test_[BATCH_SIZE].bin`, which saves `BATCH_SIZE` inputs from the test data
* `y_label_[BATCH_SIZE].bin`, the corresponding labels

## C++
To test the network with the C++ nGraph integration, change to the build directory
and run the unit test
```
cd ../../build
[NGRAPH_ENCRYPT_DATA=1] [NGRAPH_ENCRYPT_MODEL=1] [NGRAPH_BATCH_DATA=1] NGRAPH_HE_SEAL_CONFIG=../test/model/he_seal_ckks_config_N13_L7.json ./test/cryptonets_benchmark
```
This will run a pre-trained Cryptonets example on various batch sizes `N={1, 2, 4, 8, 16, ..., 4096}`.
If `NGRAPH_ENCRYPT_DATA=1`, the Cryptonets input data will be encrypted, preserving the privacy of the data.
If `NGRAPH_ENCRYPT_MODEL=1`, the Cryptonets model will be encrypted, preserving the privacy of the model.
If both `NGRAPH_ENCRPYT_DATA=1` and `NGRAPH_ENCRPYT_MODEL=1` are set, the runtime will roughly double, due to unavailibility of plaintext optimizations.
If `NGRAPH_BATCH_DATA=1`, the Cryptonets model will perform SIMD batching for greatly-increased throughput.
Note: either `NGRAPH_ENCRPYT_DATA=1` or `NGRAPH_ENCRYPT_MODEL=1` should be set; otherwise the model simply uses unencrypted computation.