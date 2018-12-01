# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

include(ExternalProject)

# ${CMAKE_CURRENT_BINARY_DIR} is he-transformer/build
set(SEAL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ext_seal)
set(SEAL_SRC_DIR ${SEAL_PREFIX}/src/ext_seal/SEAL)
set(SEAL_TAR_FILE https://download.microsoft.com/download/B/3/7/B3720F6B-4F4A-4B54-9C6C-751EF194CBE7/SEAL_3.0.tar.gz)

# TODO: remove before release
set(SEAL_SRC_DIR ${SEAL_PREFIX}/src/ext_seal/src)
set(SEAL_TAR_FILE /nfs/site/home/fboemer/repos/SEAL_3.1.tar.gz)
set(SEAL_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

ExternalProject_Add(
    ext_seal
    URL ${SEAL_TAR_FILE}
    PREFIX ${SEAL_PREFIX}
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND cmake ${SEAL_SRC_DIR}
                            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_DIR}
                            -DCMAKE_CXX_FLAGS=${SEAL_CXX_FLAGS}
                            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                            -DCMAKE_INSTALL_MESSAGE=LAZY
)

# TODO: test below for release
# SET(SEAL_REPO_URL https://github.com/Microsoft/SEAL.git)
# SET(SEAL_GIT_LABEL xxxxx) # TODO

#ExternalProject_Add(
#   ext_seal
#   GIT_REPOSITORY ${SEAL_REPO_URL}
#   GIT_TAG ${SEAL_GIT_LABEL}
#   PREFIX ${SEAL_PREFIX}
#   UPDATE_COMMAND ""
#   CONFIGURE_COMMAND cmake ${SEAL_SRC_DIR}
#   -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_DIR}
#   -DCMAKE_CXX_FLAGS=${SEAL_CXX_FLAGS}
#   -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
#   -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
#   -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
#   -DCMAKE_INSTALL_MESSAGE=LAZY
#)