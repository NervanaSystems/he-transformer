# ******************************************************************************
# Copyright 2018-2019 Intel Corporation
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

project (pyhetransformer)

include(ExternalProject)

ExternalProject_Add(
        pybind11
        GIT_REPOSITORY "https://github.com/pybind/pybind11.git"
        GIT_TAG "v2.2.4"
        SOURCE_DIR "${CMAKE_BINARY_DIR}/pybind11"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
)


set(BUILD_SH_IN "${CMAKE_SOURCE_DIR}/python/build_wheel.sh.in")
set(BUILD_SH    "${CMAKE_BINARY_DIR}/python/build_wheel.sh")
configure_file(${BUILD_SH_IN} ${BUILD_SH} @ONLY)
