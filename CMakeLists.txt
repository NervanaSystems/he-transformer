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

# To support BUILD_BYPRODUCTS
cmake_minimum_required (VERSION 3.2)

# Suppress an OS X-specific warning.
if (POLICY CMP0042)
    cmake_policy(SET CMP0042 OLD)
endif()

# These variables are undocumented but useful.
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# CXX flags
# set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z -std=gnu++1z")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

# Global project name
project (he-transformer)

# Installation paths when `make install`
if (DEFINED NGRAPH_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${NGRAPH_INSTALL_PREFIX})
else()
    set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/ngraph_dist")
endif()
set(NGRAPH_INSTALL_LIB_DIR "${CMAKE_INSTALL_PREFIX}/lib")
set(NGRAPH_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include")

# This allows libhe_backend.so to find say libntl.so in the same dir
set(CMAKE_INSTALL_RPATH "\$ORIGIN")

# HE transformer headers
set(HE_TRANSFORMER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(HE_TRANSFORMER_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test)

# External projects (ngraph, seal, heaan)
include(ExternalProject)
set(EXTERNAL_INSTALL_DIR ${CMAKE_BINARY_DIR}/external)
set(EXTERNAL_INSTALL_LIB_DIR ${EXTERNAL_INSTALL_DIR}/lib)
set(EXTERNAL_INSTALL_INCLUDE_DIR ${EXTERNAL_INSTALL_DIR}/include)
include(cmake/openmp.cmake)
include(cmake/ngraph.cmake)
include(cmake/gtest.cmake)
include(cmake/seal.cmake)
include(cmake/json.cmake)

# HE transformer source directory
add_subdirectory(src)
add_subdirectory(test)
