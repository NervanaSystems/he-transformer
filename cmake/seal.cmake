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

# ${CMAKE_CURRENT_BINARY_DIR} is ngraph/build/third-party
set(SEAL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ext_seal)
set(SEAL_SRC_DIR ${SEAL_PREFIX}/src/ext_seal/SEAL)
set(SEAL_TAR_FILE https://download.microsoft.com/download/B/3/7/B3720F6B-4F4A-4B54-9C6C-751EF194CBE7/SEAL_v2.3.0-4_Linux.tar.gz)

# TODO: if Debug, define -DSEAL_DEBUG and -DSEAL_THROW_ON_DECODER_OVERFLOW
# TODO: undefine SEAL_THROW_ON_MULTIPLY_PLAIN_BY_ZERO
message("SEAL...")

set(CXX_FLAGS -fPIC)

ExternalProject_Add(
    ext_seal
    URL ${SEAL_TAR_FILE}
    PREFIX ${SEAL_PREFIX}
    #CMAKE_ARGS "-DSEAL_DEBUG
    #           -USEAL_THROW_ON_MULTIPLY_PLAIN_BY_ZERO
    #           -DSEAL_THROW_ON_DECODER_OVERFLOW"
    CONFIGURE_COMMAND cd ${SEAL_SRC_DIR} && ./configure CXXFLAGS=${CXX_FLAGS} --prefix=${EXTERNAL_INSTALL_DIR}
    UPDATE_COMMAND ""
    BUILD_COMMAND make -j$(nproc) -C ${SEAL_SRC_DIR} CXXFLAGS+=-fPIC CXXFLAGS+=-std=c++11 CXXFLAGS+=-DSEAL_DEBUG
    #CXXFLAGS+=-USEAL_THROW_ON_MULTIPLY_PLAIN_BY_ZERO
    #CXXFLAGS+=-DSEAL_THROW_ON_DECODER_OVERFLOW
    INSTALL_COMMAND make install -C ${SEAL_SRC_DIR}
        && cp -r ${EXTERNAL_INSTALL_INCLUDE_DIR}/SEAL/seal ${EXTERNAL_INSTALL_INCLUDE_DIR}
        && rm -r ${EXTERNAL_INSTALL_INCLUDE_DIR}/SEAL
        && cp  ${EXTERNAL_INSTALL_LIB_DIR}/SEAL/libseal.a ${EXTERNAL_INSTALL_LIB_DIR}
        && rm -r ${EXTERNAL_INSTALL_LIB_DIR}/SEAL
    BUILD_ALWAYS 1
)
