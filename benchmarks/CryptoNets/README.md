This folder is used to generate runtime results for the MNIST-Cryptonets example (Table 3, Table 4, Figure 6).
Make sure the python environment with the ngraph-tf bridge is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`.

# To generate timings
## 1. `./run_timings_tf_vs_direct.sh`
This will generate runtimes output in a `results` folder. Used for Table 3 and Table 4
## 2. `./run_simd_timings.sh`
This will generate runtimes output in a `results/simd` folder. Used for Figure 6
# To analyze results
Run the jupyter notebook, `CryptonetsResults.ipynb`
