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

build()
{
   echo 'Building HE-transformer'
   mkdir build
   cd build
   cmake .. -DCMAKE_CXX_COMPILER=g++-7 -DCMAKE_C_COMPILER=gcc-7
   make -j install
   source external/venv-tf-py3/bin/activate
   cd -
}

unit_tests()
{
    echo 'Running unit-tests'
    cd build
    ./test/unit-test
    cd -
}

python_integration()
{
    echo 'Running python examples'
    cd examples
    python axpy.py
    NGRAPH_TF_BACKEND=HE_SEAL_CKKS python axpy.py
    NGRAPH_TF_BACKEND=HE_SEAL_BFV python axpy.py
    cd MNIST-Cryptonets
    NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N13_L7.json \
        NGRAPH_TF_BACKEND=HE_SEAL_CKKS \
        NGRAPH_BATCH_TF=1 \
        NGRAPH_BATCH_DATA=1 \
        NGRAPH_ENCRYPT_DATA=1 \
        python test.py --batch_size=128
}

build
unit_tests
python_integration
