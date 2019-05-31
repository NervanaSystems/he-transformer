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

SET(BOOST_ASIO_REPO_URL https://github.com/boostorg/asio)
SET(BOOST_SYSTEM_REPO_URL https://github.com/boostorg/system)
SET(BOOST_CONFIG_REPO_URL https://github.com/boostorg/config)
SET(BOOST_GIT_LABEL boost-1.69.0)

add_library(libboost INTERFACE)

ExternalProject_Add(
    ext_boost_asio
    PREFIX boost
    GIT_REPOSITORY ${BOOST_ASIO_REPO_URL}
    GIT_TAG ${BOOST_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )

ExternalProject_Get_Property(ext_boost_asio SOURCE_DIR)
message("boost asio SOURCE_DIR ${SOURCE_DIR}")
target_include_directories(libboost SYSTEM INTERFACE ${SOURCE_DIR}/include)
set(BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH} ${SOURCE_DIR})

ExternalProject_Add(
    ext_boost_system
    PREFIX boost
    GIT_REPOSITORY ${BOOST_SYSTEM_REPO_URL}
    GIT_TAG ${BOOST_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    BUILDE_IN_SOURCE 1
    )
ExternalProject_Get_Property(ext_boost_system SOURCE_DIR)
message("boost system SOURCE_DIR ${SOURCE_DIR}")
target_include_directories(libboost SYSTEM INTERFACE ${SOURCE_DIR}/include)
set(BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH} ${SOURCE_DIR})

ExternalProject_Add(
    ext_boost_config
    PREFIX boost
    GIT_REPOSITORY ${BOOST_CONFIG_REPO_URL}
    GIT_TAG ${BOOST_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )
ExternalProject_Get_Property(ext_boost_config SOURCE_DIR)
message("boost config SOURCE_DIR ${SOURCE_DIR}")
target_include_directories(libboost SYSTEM INTERFACE ${SOURCE_DIR}/include)

set(BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH} ${SOURCE_DIR})

message("BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH}")

add_dependencies(libboost ext_boost_asio ext_boost_system ext_boost_config)
