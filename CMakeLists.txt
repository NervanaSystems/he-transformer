# *****************************************************************************
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
# *****************************************************************************

cmake_minimum_required (VERSION 2.8)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# Suppress an OS X-specific warning.
if (POLICY CMP0042)
    cmake_policy(SET CMP0042 OLD)
endif()

# Global project name
project (he)

# Installation paths
if (DEFINED NGRAPH_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${NGRAPH_INSTALL_PREFIX})
else()
    set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/ngraph_dist")
endif()
set(NGRAPH_INSTALL_LIB_DIR "${CMAKE_INSTALL_PREFIX}/lib")
set(NGRAPH_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include")

# External projects (relative to HE transformer)
include(ExternalProject)
set(EXTERNAL_INSTALL_DIR ${CMAKE_BINARY_DIR}/external)

set(NGRAPH_HE_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/he_install)
set(NGRAPH_HE_INSTALL_LIB_DIR ${NGRAPH_HE_INSTALL_DIR}/lib)
set(NGRAPH_HE_INSTALL_INCLUDE_DIR ${NGRAPH_HE_INSTALL_DIR}/include)

include(external/external_gmp.cmake)
#include(external/external_ntl.cmake)
#include(external/external_heaan.cmake)

# HE transformer headers
set(HE_TRANSFORMER_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# HE transformer source directory
add_subdirectory(src)
