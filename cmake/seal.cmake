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

# ${CMAKE_CURRENT_BINARY_DIR} is he-transformer/build

set(SEAL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ext_seal)
set(SEAL_SRC_DIR ${SEAL_PREFIX}/src/ext_seal/native/src)
SET(SEAL_REPO_URL https://github.com/Microsoft/SEAL.git)
SET(SEAL_GIT_TAG origin/3.3.0)

set(SEAL_USE_CXX17 ON)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
   if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
      set(SEAL_USE_CXX17 OFF)
   endif()
endif()

message("SEAL_USE_CXX17 ${SEAL_USE_CXX17}")

ExternalProject_Add(
   ext_seal
   GIT_REPOSITORY ${SEAL_REPO_URL}
   GIT_TAG ${SEAL_GIT_TAG}
   PREFIX ${SEAL_PREFIX}
   INSTALL_DIR ${EXTERNAL_INSTALL_DIR}
   CONFIGURE_COMMAND cmake ${SEAL_SRC_DIR}
   -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_DIR}
   -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
   -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
   -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
   -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
   -DCMAKE_INSTALL_MESSAGE=LAZY
   -DSEAL_USE_CXX17=${SEAL_USE_CXX17}
)

ExternalProject_Get_Property(ext_seal SOURCE_DIR)
message("SOURCE_DIR ${SOURCE_DIR}")
add_library(libseal INTERFACE)
target_include_directories(libseal SYSTEM INTERFACE ${SOURCE_DIR}/include)
add_dependencies(libseal ext_seal)