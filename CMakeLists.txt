# *****************************************************************************
# Copyright 2018-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# *****************************************************************************

# SEAL requires >= 3.10
cmake_minimum_required(VERSION 3.10)

# Global project name
project(he_transformer LANGUAGES CXX)

# CXX flags
if(DEFINED CMAKE_CXX_CLAGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z -std=gnu++1z -Wall")
else()
  set(CMAKE_CXX_FLAGS "-std=c++1z -std=gnu++1z -Wall")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g")

set(CMAKE_INSTALL_MESSAGE LAZY)

if(CMAKE_BUILD_TYPE)
  set(RELEASE_TYPES
      Debug
      Release
      RelWithDebInfo
      MinSizeRel)
  list(FIND RELEASE_TYPES ${CMAKE_BUILD_TYPE} INDEX_FOUND)
  if(${INDEX_FOUND} EQUAL -1)
    message(
      FATAL_ERROR
        "CMAKE_BUILD_TYPE must be one of Debug, Release, RelWithDebInfo, or MinSizeRel"
      )
  endif()
endif()

include(cmake/sdl.cmake)

# These variables are undocumented but useful.
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# For ngraph/log
set(PROJECT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DPROJECT_ROOT_DIR="${PROJECT_ROOT_DIR}")

# This allows libhe_backend.so to find libraries in the same directory
set(CMAKE_INSTALL_RPATH "\$ORIGIN")

# he-transformer headers
set(HE_TRANSFORMER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# External projects (ngraph, seal, etc. )
set(EXTERNAL_INSTALL_DIR ${CMAKE_BINARY_DIR}/external)
set(EXTERNAL_INSTALL_LIB_DIR ${EXTERNAL_INSTALL_DIR}/lib)
set(EXTERNAL_INSTALL_INCLUDE_DIR ${EXTERNAL_INSTALL_DIR}/include)

if(NOT DEFINED EXTERNAL_PROJECTS_ROOT)
  set(EXTERNAL_PROJECTS_ROOT ${CMAKE_CURRENT_BINARY_DIR})
endif()

# Build options python version used in virtual environment
find_package(PythonInterp 3)
if(PYTHONINTERP_FOUND)
  message("PYTHON_VERSION_STRING " ${PYTHON_VERSION_STRING})
elseif()
  message(FATAL_ERROR "Python3 not found.")
endif()

if(NOT DEFINED PYTHON_VENV_VERSION)
  set(PYTHON_VENV_VERSION
      "python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
endif()
message("PYTHON_VENV_VERSION ${PYTHON_VENV_VERSION}")

option(NGRAPH_HE_SANITIZE_ADDRESS "Enable address sanitizer" OFF)

if(NGRAPH_HE_SANITIZE_ADDRESS)
  message(STATUS "NGRAPH_HE_SANITIZE_ADDRESS")
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer")
endif()

include(cmake/ngraph-tf.cmake)
include(cmake/gtest.cmake)
include(cmake/json.cmake)
include(cmake/openmp.cmake)
include(cmake/seal.cmake)
include(cmake/boost.cmake)

message("ngraph NGRAPH_TF_INCLUDE_DIR ${NGRAPH_TF_INCLUDE_DIR}")

# Add ngraph library
add_library(ngraph SHARED IMPORTED)
set_target_properties(ngraph
                      PROPERTIES IMPORTED_LOCATION
                                 ${NGRAPH_TF_LIB_DIR}/libngraph.so)
set_target_properties(ngraph
                      PROPERTIES INCLUDE_DIRECTORIES ${NGRAPH_TF_INCLUDE_DIR})

# HE transformer source and test directories
add_subdirectory(src)
add_subdirectory(test)

# For python bindings
add_subdirectory(python)
