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

set(NTL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ext_ntl)
set(NTL_SOURCE_DIR ${NTL_PREFIX}/src/ext_ntl)

ExternalProject_Add(
    ext_ntl
    DEPENDS ext_gmp
    DOWNLOAD_COMMAND wget http://www.shoup.net/ntl/ntl-10.5.0.tar.gz
    COMMAND tar -xzf ntl-10.5.0.tar.gz -C ${NTL_SOURCE_DIR} --strip 1
    COMMAND rm ntl-10.5.0.tar.gz
    PREFIX ${NTL_PREFIX}
    CONFIGURE_COMMAND cd ${NTL_SOURCE_DIR}/src && ./configure NTL_GMP_LIP=on SHARED=on PREFIX=${EXTERNAL_INSTALL_DIR} GMP_PREFIX=${EXTERNAL_INSTALL_DIR}
    BUILD_COMMAND make -j$(nproc) -C ${NTL_SOURCE_DIR}/src
    INSTALL_COMMAND make install -C ${NTL_SOURCE_DIR}/src
)
