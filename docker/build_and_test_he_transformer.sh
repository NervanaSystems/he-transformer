#!  /bin/bash

set -e
# set -u  # Cannot use set -u, as activate below relies on unbound variables
set -o pipefail

# If proxy settings are detected in the environment, make sure they are
# included on the docker-build command-line.
if [ ! -z "${http_proxy}" ] ; then
    DOCKER_HTTP_PROXY="--build-arg http_proxy=${http_proxy}"
else
    DOCKER_HTTP_PROXY=' '
fi

if [ ! -z "${https_proxy}" ] ; then
    DOCKER_HTTPS_PROXY="--build-arg https_proxy=${https_proxy}"
else
    DOCKER_HTTPS_PROXY=' '
fi

uname -a
cat /etc/os-release || true

echo ' '
echo 'Contents of /home:'
ls -la /home
echo ' '

export HE_SRC_DIR=/home

echo 'Argument'
echo $1
echo $2
echo 'Done shows args'

git pull
git checkout fboemer/ngraph-tf-integration1



build_he_transformer()
{
    cd $HE_SRC_DIR
    echo 'Building HE Transformer'
    rm -rf build
    mkdir build
    cd build
    cmake .. -DCMAKE_CXX_COMPILER=g++-7 -DCMAKE_C_COMPILER=gcc-7
    make -j install

    source external/venv-tf-py3/bin/activate
    cd $HE_SRC_DIR
}

# Test C++ integration
run_unit_tests()
{
    echo 'Running unit-tests'
    cd $HE_SRC_DIR/build
    ./test/unit-test
    # Cryptonets tests
    # Encrypt data
    NGRAPH_TF_BACKEND=HE_SEAL_CKKS NGRAPH_BATCH_DATA=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_HE_SEAL_CONFIG=../test/model/he_seal_ckks_config_13.json ./test/cryptonets_benchmark --gtest_filter="Cryptonets.CKKS_4096"
    # Encrypt model
    NGRAPH_TF_BACKEND=HE_SEAL_CKKS NGRAPH_BATCH_DATA=1 NGRAPH_ENCRYPT_MODEL=1 NGRAPH_HE_SEAL_CONFIG=../test/model/he_seal_ckks_config_13.json ./test/cryptonets_benchmark --gtest_filter="Cryptonets.CKKS_4096"

    echo "Done running unit-tests"
    cd $HE_SRC_DIR
}

run_python_tests()
{
    cd $HE_SRC_DIR/examples
    NGRAPH_TF_BACKEND=HE_SEAL_BFV python axpy.py
    NGRAPH_TF_BACKEND=HE_SEAL_CKKS python axpy.py
    cd $HE_SRC_DIR
}

run_cryptonets_tests()
{
    cd $HE_SRC_DIR/examples/cryptonets

    # Test cryptonets under python
    NGRAPH_TF_BACKEND=HE_SEAL_CKKS NGRAPH_BATCH_DATA=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_BATCH_TF=1 NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_13.json python test.py --batch_size=4096 --report_accuracy=1
    NGRAPH_TF_BACKEND=HE_SEAL_CKKS NGRAPH_BATCH_DATA=1 NGRAPH_ENCRYPT_MODEL=1 NGRAPH_BATCH_TF=1 NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_13.json python test.py --batch_size=4096 --report_accuracy=1
}


build_he_transformer &&
run_unit_tests &&
run_python_tests &&
run_cryptonets_tests