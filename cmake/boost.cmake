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

include(ExternalProject)

set(BOOST_REPO_URL https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.gz)

ExternalProject_Add(
    ext_boost
    PREFIX boost
    URL ${BOOST_REPO_URL}
    URL_HASH SHA256=9a2c2819310839ea373f42d69e733c339b4e9a19deab6bfec448281554aa4dbb
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )

ExternalProject_Get_Property(ext_boost SOURCE_DIR)

set(BOOST_HEADERS_PATH ${SOURCE_DIR})
message(STATUS "BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH}")

add_library(libboost INTERFACE)
add_dependencies(libboost ext_boost)
target_include_directories(libboost SYSTEM INTERFACE ${BOOST_HEADERS_PATH})