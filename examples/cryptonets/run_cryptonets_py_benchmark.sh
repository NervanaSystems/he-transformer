#!/usr/bin/env bash

for i in `seq 10`; do
    NGRAPH_HE_HEAAN_CONFIG=heaan_config_13.json NGRAPH_TF_BACKEND=HE_HEAAN python run_cryptonets_py_benchmark;
    NGRAPH_HE_HEAAN_CONFIG=heaan_config_14.json NGRAPH_TF_BACKEND=HE_HEAAN python run_cryptonets_py_benchmark;
done
