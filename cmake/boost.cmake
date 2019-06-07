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

SET(BOOST_REPO_URL https://github.com/boostorg/boost)
SET(BOOST_GIT_LABEL boost-1.69.0)

ExternalProject_Add(
    ext_boost
    PREFIX boost
    GIT_REPOSITORY ${BOOST_REPO_URL}
    GIT_TAG ${BOOST_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )

ExternalProject_Get_Property(ext_boost SOURCE_DIR)
message("boost SOURCE_DIR ${SOURCE_DIR}")
set(BOOST_LIB_DIR ${SOURCE_DIR}/libs)

set(BOOST_LIBS
asio
system
config
throw_exception
detail
assert
date_time
smart_ptr
core
predef
utility
type_traits
static_assert
mpl
preprocessor
numeric_conversion
bind
regex)

foreach(BOOST_LIB ${BOOST_LIBS})
    set(BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH} ${BOOST_LIB_DIR}/${BOOST_LIB}/include)
endforeach()

message("BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH}")

#include_directories(${BOOST_HEADERS_PATH})
add_library(libboost INTERFACE)
add_dependencies(libboost ext_boost)
target_include_directories(libboost SYSTEM INTERFACE ${BOOST_HEADERS_PATH})

