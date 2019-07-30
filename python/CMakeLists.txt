# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
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
ExternalProject_Add(
        pybind11
        GIT_REPOSITORY "https://github.com/pybind/pybind11.git"
        GIT_TAG "v2.2.4"
        SOURCE_DIR "${CMAKE_BINARY_DIR}/pybind11"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
)

message("python CMAKE_SOURCE_DIR ${CMAKE_SOURCE_DIR}")
message("python CMAKE_BINARY_DIR ${CMAKE_BINARY_DIR}")
message("python CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}")

set(BUILD_SH_IN "${CMAKE_SOURCE_DIR}/python/build_wheel.sh.in")
set(BUILD_SH    "${CMAKE_BINARY_DIR}/python/build_wheel.sh")
configure_file(${BUILD_SH_IN} ${BUILD_SH} @ONLY)

message("EXTERNAL_INSTALL_LIB_DIR ${EXTERNAL_INSTALL_LIB_DIR}")

set(PYTHON_ARTIFACTS_DIR ${CMAKE_BINARY_DIR}/python/_install)
message("PYTHON_ARTIFACTS_DIR ${PYTHON_ARTIFACTS_DIR}")
set(PYTHON_LIB_DIR ${PYTHON_ARTIFACTS_DIR}/lib)
set(PYTHON_INCLUDE_DIR ${PYTHON_ARTIFACTS_DIR}/include)

message("PROJECT_ROOT_DIR ${PROJECT_ROOT_DIR}")

# Hack to install files for python wheel
# When specifying DESTDIR in custom command, this will change the destination
# to DESTDIR, rather than EXTERNAL_INSTALL_INCLUDE_DIR
install(DIRECTORY ${EXTERNAL_INSTALL_INCLUDE_DIR}/
        DESTINATION ${EXTERNAL_INSTALL_INCLUDE_DIR}
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h")
add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/python/dist/
        POST_BUILD
        DEPENDS he_seal_backend
        #WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND VERBOSE=1 make -C ../ DESTDIR=python/_install install && bash build_wheel.sh
)

add_custom_target(python_wheel DEPENDS he_seal_backend
                                       ${CMAKE_BINARY_DIR}/python/dist/)