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
set(SEAL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/seal)
set(SEAL_SRC_DIR ${SEAL_PREFIX}/src/external_seal/SEAL)
set(SEAL_TAR_FILE ${CMAKE_CURRENT_SOURCE_DIR}/third-party/SEAL_v2.3.0-4_Linux.tar.gz)

ExternalProject_Add(
    external_seal
    URL ${SEAL_TAR_FILE}
    PREFIX ${SEAL_PREFIX}
    CONFIGURE_COMMAND cd ${SEAL_SRC_DIR} && ./configure --prefix=${NGRAPH_HE_INSTALL_DIR}
    BUILD_COMMAND make -j$(nproc) -C ${SEAL_SRC_DIR}
    INSTALL_COMMAND make install -C ${SEAL_SRC_DIR}
    BUILD_ALWAYS 1
)
