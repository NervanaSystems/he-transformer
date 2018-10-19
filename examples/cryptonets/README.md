This example demonstrates the [Cryptonets](https://www.microsoft.com/en-us/research/publication/cryptonets-applying-neural-networks-to-encrypted-data-with-high-throughput-and-accuracy/) network, which achieves ~99% accuracy on MNIST.

This example depends on the [ngraph-tensorflow bridge](https://github.com/NervanaSystems/ngraph-tensorflow-bridge/). Make sure the python environment with ng-tf bridge is active, i.e. run `source ~/repos/venvs/he3/bin/activate`.

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
[NGRAHP_HE_HEAAN_CONFIG = heaan_config_1[X].json] NGRAPH_TF_BACKEND=HE:HEAAN python test.py
```

This runs inference on the Cryptonets network using the HEAAN backend.

For optimal performance, install Open MP and call the commands with `OMP_NUM_THREADS=$(nproc)$`, i.e.
```
OMP_NUM_THREADS=$(nproc)$` NGRAPH_TF_BACKEND=HE:HEAAN python test.py
```

To export the serialized model for use in C++ integration with nGraph, run
```
NGRAPH_ENABLE_SERIALIZE=1 python test.py --save_batch=1 [--batch_size=BATCH_SIZE]
```

This will generate:
* `mnist_cryptonets_batch_[BATCH_SIZE].json`, which is the serialized nGraph computation graph.
* `x_test_[BATCH_SIZE].bin`, which saves `BATCH_SIZE` inputs from the test data
* `y_label_[BATCH_SIZE].bin`, the corresponding labels

## C++
To test the network with the C++ nGraph integration, copy these files to the unit-tests,
```
cp mnist_cryptonets_batch_[BATCH_SIZE].json ../../test/model
cp x_test_[BATCH_SIZE].bin ../../test/model
cp y_label_[BATCH_SIZE].bin ../../test/model
```
and run the unit test
```
cd ../../build
./test/unit-test --gtest_filter="HE_HEAAN.cryptonets_benchmark_heaan_N"
```
for `N` a power of two in `{1, 2, 4, 8, ..., 4096}`

For optimal performance, run with
```
NGRAPH_HE_HEAAN_CONFIG=model/config_13.json ./test/unit-test --gtest_filter="HE_HEAAN.cryptonets_benchmark_heaan_N"
```

