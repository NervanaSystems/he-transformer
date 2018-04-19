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

set(HEAAN_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ext_heaan)
set(HEAAN_SOURCE_DIR ${HEAAN_PREFIX}/src/ext_heaan)

ExternalProject_Add(
    ext_heaan
    DEPENDS ext_ntl
    GIT_REPOSITORY https://github.com/NervanaSystems/HEAAN.git
    GIT_TAG master
    PREFIX ${HEAAN_PREFIX}
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_INSTALL_PREFIX=${NGRAPH_HE_INSTALL_DIR}
        -DNGRAPH_HE_INSTALL_INCLUDE_DIR=${NGRAPH_HE_INSTALL_INCLUDE_DIR}
        -DNGRAPH_HE_INSTALL_LIB_DIR=${NGRAPH_HE_INSTALL_LIB_DIR}
)
