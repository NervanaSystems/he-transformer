#!  /bin/bash

set -e
# set -u  # Cannot use set -u, as activate below relies on unbound variables
set -o pipefail

# Debugging to verify builds on CentOS 7.4 and Ubuntu 16.04
if [ -f "/etc/centos-release" ]; then
    cat /etc/centos-release
fi

if [ -f "/etc/lsb-release" ]; then
    cat /etc/lsb-release
fi

uname -a
cat /etc/os-release || true

if [ ! -z "${http_proxy}" ] ; then
    DOCKER_HTTP_PROXY="--build-arg http_proxy=${http_proxy}"
else
    DOCKER_HTTP_PROXY=' '
fi

echo 'https proxy' ${https_proxy}
echo 'http proxy' ${http_proxy}

if [ ! -z "${https_proxy}" ] ; then
    DOCKER_HTTPS_PROXY="--build-arg https_proxy=${https_proxy}"
else
    DOCKER_HTTPS_PROXY=' '
fi

DFILE="Dockerfile.ngraph-he"

CONTEXT='.'

# build the docker base image
docker build  --rm=true \
       ${DOCKER_HTTP_PROXY} ${DOCKER_HTTPS_PROXY} \
       -f="${DFILE}" \
       ${CONTEXT}

    echo 'Docker image build completed'

