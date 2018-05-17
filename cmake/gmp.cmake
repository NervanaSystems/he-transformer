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

# NTL depends on GMP
set(GMP_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ext_gmp)
set(GMP_SOURCE_DIR ${GMP_PREFIX}/src/ext_gmp)
message("GMP_PREFIX " ${GMP_PREFIX})
message("EXTERNAL_INSTALL_DIR " ${EXTERNAL_INSTALL_DIR})

set(GMP_INCLUDE_DIR ${GMP_SOURCE_DIR}/include)
set(GMP_LIB_DIR ${GMP_SOURCE_DIR}/lib)

ExternalProject_Add(
    ext_gmp
    DOWNLOAD_COMMAND wget https://ftp.gnu.org/gnu/gmp/gmp-6.1.2.tar.xz && tar xfJ gmp-6.1.2.tar.xz --strip 1
    PREFIX ${GMP_PREFIX}
    CONFIGURE_COMMAND
    cd ${GMP_SOURCE_DIR} && ./configure --prefix=${GMP_PREFIX}
    UPDATE_COMMAND ""
    BUILD_COMMAND make -j$(nproc) -C ${GMP_SOURCE_DIR}
    INSTALL_COMMAND make -j$(nproc) install -C ${GMP_SOURCE_DIR}
)
