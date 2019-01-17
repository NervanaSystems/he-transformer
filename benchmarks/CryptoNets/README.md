This folder is used to generate runtime results for the MNIST-Cryptonets example.
Make sure the python environment with the ngraph-tf bridge is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`.

# To generate timings
## 1. `./run_timings_tf_vs_direct.sh`
This will generate tuntimes output in a `results` folder
## 2. `./run_simd_timings.sh`
This will generate runtimt output in a `results/sim` folder
# To analyze results
Run the jupyter notebook, `CryptonetsResults`
