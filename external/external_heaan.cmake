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
set(CFLAGS "-g -O2 -std=c++11 -pthread -DFHE_THREADS -DFHE_BOOT_THREADS -fmax-errors=2 -fPIC -I${NGRAPH_HE_INSTALL_INCLUDE_DIR}")

ExternalProject_Add(
    ext_heaan
    DEPENDS ext_ntl
    GIT_REPOSITORY https://github.com/kimandrik/HEAAN.git
    GIT_TAG master
    PREFIX ${HEAAN_PREFIX}
    CONFIGURE_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND
    COMMAND echo "HETEST" && echo
        "
        set(LIB_NAME heaan)
        project(HEAAN)
        set (CMAKE_CXX_STANDARD 11)
        include_directories(src)
        file(GLOB SOURCES \"src/*.cpp\") # TODO: remove heaan.cpp
        find_library(NTL_LIB ntl)
        find_library(GMP_LIB gmp)
        target_link_libraries(heaan ${NTL_LIB} ${M_LIB} ${GMP_LIB})
        add_library(heaan STATIC ${SOURCES})
        " > ${HEAAN_PREFIX}/CMakeLists.txt
    #COMMAND mkdir -p ${HEAAN_PREFIX}/build
    #COMMAND cd ${HEAAN_PREFIX}/build
    BUILD_COMMAND "" $cmake .. && make -j$(nproc)
    BUILD_ALWAYS 1
)
