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

# These variables are undocumented but useful.
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# CXX flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z -std=gnu++1z")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g")
set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")

set(CMAKE_INSTALL_MESSAGE LAZY)

include(cmake/sdl.cmake)

# Global project name
project (he-transformer)

add_definitions(-DPROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# Installation paths when `make install`
if (DEFINED NGRAPH_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${NGRAPH_INSTALL_PREFIX})
else()
    set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/ngraph_dist")
endif()
set(NGRAPH_INSTALL_LIB_DIR "${CMAKE_INSTALL_PREFIX}/lib")
set(NGRAPH_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include")

# This allows libhe_backend.so to find libraries in the same directory
set(CMAKE_INSTALL_RPATH "\$ORIGIN")

option(ENABLE_TF "Support Tensorflow integration through ngraph-tf" OFF)

message("ENABLE_TF ${ENABLE_TF}")

# Check python environment is active if TF is enabled
if (ENABLE_TF)
    execute_process(
        COMMAND python -c "import os; print(os.environ['VIRTUAL_ENV'])"
        RESULT_VARIABLE PY_ENV_ERROR
    )
    if(PY_ENV_ERROR)
        message(FATAL_ERROR "Set python virtual environment when building with TF support.")
    endif()
endif()

# he-transformer headers
set(HE_TRANSFORMER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(HE_TRANSFORMER_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test)

# External projects (ngraph, seal, etc. )
set(EXTERNAL_INSTALL_DIR ${CMAKE_BINARY_DIR}/external)
set(EXTERNAL_INSTALL_LIB_DIR ${EXTERNAL_INSTALL_DIR}/lib)
set(EXTERNAL_INSTALL_INCLUDE_DIR ${EXTERNAL_INSTALL_DIR}/include)

if(NOT DEFINED EXTERNAL_PROJECTS_ROOT)
    set(EXTERNAL_PROJECTS_ROOT ${CMAKE_CURRENT_BINARY_DIR})
endif()

include(cmake/gtest.cmake)
include(cmake/json.cmake)
include(cmake/ngraph.cmake)

if (ENABLE_TF)
    message("Building with Tensorflow integration through ngraph-tf")

    # Allows "make ext_ngraph_tf"
    add_custom_target(
        install_tensorflow
        COMMAND pip install -U tensorflow
    )
    include(cmake/ngraph-tf.cmake)
endif() # ENABLE_TF

include(cmake/openmp.cmake)
include(cmake/seal.cmake)

# HE transformer source and test directories
add_subdirectory(src)
add_subdirectory(test)
