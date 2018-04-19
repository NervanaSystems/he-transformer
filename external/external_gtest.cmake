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

# Enable ExternalProject CMake module
include(ExternalProject)
set(GTEST_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ext_gtest)

#----------------------------------------------------------------------------------------------------------
# Download and install GoogleTest ...
#----------------------------------------------------------------------------------------------------------

SET(GTEST_GIT_REPO_URL https://github.com/google/googletest.git)
SET(GTEST_GIT_LABEL release-1.8.0)

# The 'BUILD_BYPRODUCTS' argument was introduced in CMake 3.2.
ExternalProject_Add(
    ext_gtest
    GIT_REPOSITORY ${GTEST_GIT_REPO_URL}
    GIT_TAG ${GTEST_GIT_LABEL}
    # Disable install step
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_FLAGS="-fPIC"
    TMP_DIR "${GTEST_PREFIX}/tmp"
    STAMP_DIR "${GTEST_PREFIX}/stamp"
    DOWNLOAD_DIR "${GTEST_PREFIX}/download"
    SOURCE_DIR "${GTEST_PREFIX}/src"
    BINARY_DIR "${GTEST_PREFIX}/build"
    INSTALL_DIR "${GTEST_PREFIX}"
    BUILD_BYPRODUCTS "${GTEST_PREFIX}/build/googlemock/gtest/libgtest.a"
)

#----------------------------------------------------------------------------------------------------------

get_filename_component(
    GTEST_INCLUDE_DIR
    "${GTEST_PREFIX}/src/googletest/include"
    ABSOLUTE)

# Create a libgtest target to be used as a dependency by test programs
add_library(libgtest IMPORTED STATIC GLOBAL)
add_dependencies(libgtest ext_gtest)

# Set libgtest properties
set_target_properties(libgtest PROPERTIES
    "IMPORTED_LOCATION" "${GTEST_PREFIX}/build/googlemock/gtest/libgtest.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)
