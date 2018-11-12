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

# Test C++ integration
test_cpp_build ()
{
    echo 'Testing C++ build'
    rm -rf build
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j22
    ./test/unit-test
    echo 'Testing cryptonets'
    NGRAPH_HE_SEAL_CONFIG=../test/model/he_seal_ckks_config_13.json ./test/cryptonets_benchmark --gtest_filter="Cryptonets.CKKS_4096"
    NGRAPH_HE_SEAL_CONFIG=../test/model/he_seal_ckks_config_14.json ./test/cryptonets_benchmark --gtest_filter="Cryptonets.CKKS_4096"

    echo 'Done testing C++ build.'
}

# Test python integration
test_python_build ()
{
    echo 'Testing python build'
    cd ..
    rm -rf build
    mkdir build
    mkdir -p ~/venvs
    python -m virtualenv venv
    virtualenv ~/venvs/he3 -p python3
    source ~/venvs/he3/bin/activate
    cd build
    cmake .. -DENABLE_TF=on -DCMAKE_BUILD_TYPE=Release
    make -j22
    make install

    # Test c++ unit-tests under python
    ./test/unit-test

    # Test python unit-test
    cd ../examples
    NGRAPH_TF_BACKEND=HE:SEAL:BFV python axpy.py
    NGRAPH_TF_BACKEND=HE:SEAL:CKKS python axpy.py

    # Test cryptonets under python
    cd cryptonets
    NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_13.json NGRAPH_TF_BACKEND=HE:SEAL:CKKS python test.py --batch_size=1
    NGRAPH_BATCHED_TENSOR=1 NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_13.json NGRAPH_TF_BACKEND=HE:SEAL:CKKS python test.py --batch_size=4096

    echo 'Done testing python build'
}

test_cpp_build
test_python_build


