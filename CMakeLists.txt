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

# Cmake flags SEAL requires C++17
set(NGRAPH_HE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD ${NGRAPH_HE_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_INSTALL_MESSAGE LAZY)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Create compilation database compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(NGRAPH_HE_FORWARD_CMAKE_ARGS
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD}
    -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=${CMAKE_CXX_STANDARD_REQUIRED}
    -DCMAKE_CXX_EXTENSIONS:BOOL=${CMAKE_CXX_EXTENSIONS}
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=${CMAKE_EXPORT_COMPILE_COMMANDS}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=${CMAKE_POSITION_INDEPENDENT_CODE}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})

# These variables are undocumented but useful.
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
  message(STATUS "Setting clang flags...")
  include(cmake/clang_flags.cmake)
endif()

include(cmake/sdl.cmake)

# For ngraph/log
set(PROJECT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DPROJECT_ROOT_DIR="${PROJECT_ROOT_DIR}")

# he-transformer headers
set(HE_TRANSFORMER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# External projects (ngraph, seal, etc. )
set(EXTERNAL_INSTALL_DIR ${CMAKE_BINARY_DIR}/external)
set(EXTERNAL_INSTALL_LIB_DIR ${EXTERNAL_INSTALL_DIR}/lib)
set(EXTERNAL_INSTALL_INCLUDE_DIR ${EXTERNAL_INSTALL_DIR}/include)

set(NGRAPH_HE_INSTALL_DOC ${CMAKE_BINARY_DIR}/doc)

if(NOT DEFINED EXTERNAL_PROJECTS_ROOT)
  set(EXTERNAL_PROJECTS_ROOT ${CMAKE_CURRENT_BINARY_DIR})
endif()

# Build options python version used in virtual environment
find_package(PythonInterp 3)
if(NOT PYTHONINTERP_FOUND)
  message(FATAL_ERROR "Python3 not found.")
endif()

if(NOT DEFINED PYTHON_VENV_VERSION)
  set(PYTHON_VENV_VERSION
      "python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
endif()

option(NGRAPH_HE_SANITIZE_ADDRESS "Enable address sanitizer" OFF)
option(NGRAPH_HE_DOC_BUILD_ENABLE "Enable building of documentation" OFF)
option(NGRAPH_HE_CLANG_TIDY "Enable clang-tidy checks" ON)

# Print options
message(STATUS "NGRAPH_HE_CXX_STANDARD:     ${NGRAPH_HE_CXX_STANDARD}")
message(STATUS "NGRAPH_HE_DOC_BUILD_ENABLE: ${NGRAPH_HE_DOC_BUILD_ENABLE}")
message(STATUS "NGRAPH_HE_CLANG_TIDY:       ${NGRAPH_HE_CLANG_TIDY}")
message(STATUS "PYTHON_VENV_VERSION:        ${PYTHON_VENV_VERSION}")
message(STATUS "PYTHON_VERSION_STRING:      ${PYTHON_VERSION_STRING}")

if(NGRAPH_HE_SANITIZE_ADDRESS)
  message(STATUS "NGRAPH_HE_SANITIZE_ADDRESS")
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer")
endif()

if(NGRAPH_HE_CLANG_TIDY)
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "CMake_RUN_CLANG_TIDY requires an out-of-source build!")
  endif()
  find_program(CLANG_TIDY_COMMAND NAMES clang-tidy)
  if(NOT CLANG_TIDY_COMMAND)
    message(
      FATAL_ERROR "CMake_RUN_CLANG_TIDY is ON but clang-tidy is not found!")
  endif()
  set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")

  # Create a preprocessor definition that depends on .clang-tidy content so the
  # compile command will change when .clang-tidy changes.  This ensures that a
  # subsequent build re-runs clang-tidy on all sources even if they do not
  # otherwise need to be recompiled.  Nothing actually uses this definition.  We
  # add it to targets on which we run clang-tidy just to get the build
  # dependency on the .clang-tidy file.
  file(SHA1 ${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy clang_tidy_sha1)
  set(CLANG_TIDY_DEFINITIONS "CLANG_TIDY_SHA1=${clang_tidy_sha1}")
  unset(clang_tidy_sha1)
endif()
configure_file(.clang-tidy .clang-tidy COPYONLY)

# ------------------------------------------------------------------------------
# Installation logic...
# ------------------------------------------------------------------------------

if(LINUX)
  include(GNUInstallDirs)
else()
  set(CMAKE_INSTALL_BINDIR "bin")
  set(CMAKE_INSTALL_INCLUDEDIR "include")
  set(CMAKE_INSTALL_DOCDIR "doc")
  set(CMAKE_INSTALL_LIBDIR "lib")
endif()

if(DEFINED NGRAPH_HE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX ${NGRAPH_HE_INSTALL_PREFIX})
endif()
message(STATUS "Installation directory: ${CMAKE_INSTALL_PREFIX}")

# Destinations
set(NGRAPH_HE_INSTALL_LIB "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
set(NGRAPH_HE_INSTALL_INCLUDE
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
set(NGRAPH_HE_INSTALL_DOC "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DOCDIR}")
set(NGRAPH_HE_INSTALL_BIN "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}")

# ------------------------------------------------------------------------------
# Compiler-specific logic...
# ------------------------------------------------------------------------------

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wall -march=native -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g")

# This allows libhe_seal_backend.so to find libraries in the same directory
set(CMAKE_INSTALL_RPATH "\$ORIGIN")

# Get OS version
if(NOT APPLE)
  execute_process(COMMAND cat /etc/os-release
                  OUTPUT_VARIABLE LSB_RELEASE_ID_SHORT
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REPLACE "\""
                 ""
                 LSB_RELEASE_ID_SHORT
                 ${LSB_RELEASE_ID_SHORT})
  string(REGEX MATCH
               "ID=\([a-z])+"
               OS_VERSION
               "${LSB_RELEASE_ID_SHORT}")
  string(REGEX MATCH
               "([a-z])+"
               OS_VERSION
               "${OS_VERSION}")
  message(STATUS "OS version: ${OS_VERSION}")
endif()

if(UNIX OR APPLE)
  add_custom_target(
    style-check
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/maint/check-code-format.sh)
  add_custom_target(
    style-apply
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/maint/apply-code-format.sh)
  add_custom_target(
    style
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/maint/apply-code-format.sh)
endif()

# ------------------------------------------------------------------------------
# Print Global Options
# ------------------------------------------------------------------------------
message(STATUS "NGRAPH_HE_FORWARD_CMAKE_ARGS ${NGRAPH_HE_FORWARD_CMAKE_ARGS}")
message(STATUS "Compile Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Shared Link Flags: ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")

include(cmake/ngraph-tf.cmake)

# Install he-transformer to installation location and into ngraph-tf
set(INSTALL_LIB_DIRS
    ${EXTERNAL_INSTALL_LIB_DIR}
    ${NGRAPH_TF_VENV_LIB_DIR}
    ${NGRAPH_TF_LIB_DIR}
    ${NGRAPH_HE_INSTALL_LIB})

set(INSTALL_INCLUDE_DIRS ${EXTERNAL_INSTALL_INCLUDE_DIR}
                         ${NGRAPH_HE_INSTALL_INCLUDE})

include(cmake/gtest.cmake)
include(cmake/json.cmake)
include(cmake/openmp.cmake)
include(cmake/seal.cmake)
include(cmake/boost.cmake)
include(cmake/protobuf.cmake)

# HE transformer source and test directories
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(doc)

# For python bindings
add_subdirectory(python)
