#!  /bin/bash

set -e
# set -u  # Cannot use set -u, as activate below relies on unbound variables
set -o pipefail

if [ -z $DOCKER_TAG ]; then
    DOCKER_TAG=build_ngraph_he
fi

if [ -z $DOCKER_IMAGE_NAME ]; then
    DOCKER_IMAGE_NAME=${DOCKER_TAG}
fi

# Debugging to verify builds on CentOS 7.4 and Ubuntu 16.04
if [ -f "/etc/centos-release" ]; then
    cat /etc/centos-release
fi

if [ -f "/etc/lsb-release" ]; then
    cat /etc/lsb-release
fi

uname -a
cat /etc/os-release || true

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

DFILE="Dockerfile.he-transformer"
DIMAGE_NAME="${DOCKER_IMAGE_NAME}"
DIMAGE_VERSION=`date -Iseconds | sed -e 's/:/-/g'`

DIMAGE_ID="${DIMAGE_NAME}:${DIMAGE_VERSION}"

echo "DOCKER_HTTP_PROXY ${DOCKER_HTTP_PROXY}"
echo "DOCKER_HTTPS_PROXY ${DOCKER_HTTPS_PROXY}"

# Set docker to he-transformer root directory
CONTEXT='..'

# build the docker base image
docker build  --rm=true \
       ${GITHUB_USER} ${GITHUB_TOKEN} ${DOCKER_HTTP_PROXY} ${DOCKER_HTTPS_PROXY}  \
       -f="${DFILE}" \
       -t="${DIMAGE_ID}" \
       ${CONTEXT}

docker tag "${DIMAGE_ID}"  "${DIMAGE_NAME}:latest"

echo 'Docker image build completed'