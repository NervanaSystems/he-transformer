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




#find_package(Boost 1.69)
#if(Boost_FOUND)
#  message("Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIRS}")
#  include_directories(${Boost_INCLUDE_DIRS})
#else()
#  message(FATAL_ERROR "Boost not found")
#endif()
#add_library(boost INTERFACE)