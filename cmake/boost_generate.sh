#!/bin/bash

libraries=$*

HEADER='# ******************************************************************************
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
'

LIBRARY="SET(BOOST_GIT_LABEL boost-1.69.0)

add_library(libboost INTERFACE)"

function to_upper() {
  echo $1 | tr '[a-z]' '[A-Z]'
}

function repo_template() {
  echo "SET(BOOST_$(to_upper $1)_REPO_URL https://github.com/boostorg/$1)"
}

function build_source() {
  if [ $1 = system ]; then 
    echo "BUILDE_IN_SOURCE 1
    "
  fi
}

function project_template() {
  echo "ExternalProject_Add(
    ext_boost_$1
    PREFIX boost
    GIT_REPOSITORY \${BOOST_$(to_upper $1)_REPO_URL}
    GIT_TAG \${BOOST_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND \"\"
    BUILD_COMMAND \"\"
    INSTALL_COMMAND \"\"
    UPDATE_COMMAND \"\"
    EXCLUDE_FROM_ALL TRUE
    $(build_source $1))
ExternalProject_Get_Property(ext_boost_$1 SOURCE_DIR)
message(\"boost $1 SOURCE_DIR \${SOURCE_DIR}\")
target_include_directories(libboost SYSTEM INTERFACE \${SOURCE_DIR}/include)
set(BOOST_HEADERS_PATH \${BOOST_HEADERS_PATH} \${SOURCE_DIR})
"
 
}

FOOTER='message("BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH}")

add_dependencies(libboost'

echo "${HEADER}"
for lib in ${libraries}; do
  repo_template $lib
done
echo "${LIBRARY}
"
for lib in ${libraries}; do
  project_template $lib
done
echo -n "${FOOTER}"
for lib in ${libraries}; do
  echo -n " ext_boost_${lib}"
done
echo ')'

