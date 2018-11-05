This example demonstrates the [Cryptonets](https://www.microsoft.com/en-us/research/publication/cryptonets-applying-neural-networks-to-encrypted-data-with-high-throughput-and-accuracy/) network, which achieves ~99% accuracy on MNIST.

Note: performance is greatly improved by use of parallelism. Make sure OpenMP is installed and utilizing available cores. If you run out of memory, or the test takes too long, you are better off only running the C++ unit-tests `./test/unit-test` from the build folder instead.

With `OMP_NUM_THREADS=4` and using the smaller of two parameter settings, `he_seal_config_13.json`, the model requires about 45GB of memory.

This example depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/NervanaSystems/ngraph-tf). Make sure the python environment with ngraph-tf bridge is active, i.e. run `source ~/repos/venvs/he3/bin/activate`.

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
NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_13.json NGRAPH_TF_BACKEND=HE:SEAL:CKKS python test.py --batch_size=1
```
This runs inference on the Cryptonets network using the SEAL CKKS backend.
The `he_seal_ckks_config_13.json` file specifies the parameters which to run the model on. You can also use the `he_seal_ckks_config_14.json` or create your own configuration. Note: the model currently doesn't support batch sizes besides 1.

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
NGRAPH_HE_SEAL_CONFIG=../test/model/he_seal_ckks_config_13.json ./test/cryptonets_benchmark
```
This will run a pre-trained Cryptonets example on various batch sizes `N={1, 2, 4, 8, 16, ..., 4096}`.
