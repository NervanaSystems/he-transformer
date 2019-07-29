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
set(SEAL_REPO_URL https://github.com/Microsoft/SEAL.git)
set(SEAL_GIT_TAG origin/3.3.1)

set(SEAL_USE_CXX17 ON)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
   if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
      set(SEAL_USE_CXX17 OFF)
      message("turning SEAL_USE_CXX17 off")
   endif()
endif()

message(STATUS "SEAL_USE_CXX17 ${SEAL_USE_CXX17}")

# Without these, SEAL's globals.cpp will be deallocated twice, once by
# he_seal_backend, which loads libseal.a, and once by the global destructor.
set(SEAL_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")

ExternalProject_Add(
   ext_seal
   GIT_REPOSITORY ${SEAL_REPO_URL}
   GIT_TAG ${SEAL_GIT_TAG}
   PREFIX ${SEAL_PREFIX}
   INSTALL_DIR ${EXTERNAL_INSTALL_DIR}
   CONFIGURE_COMMAND cmake ${SEAL_SRC_DIR}
   -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_DIR}
   -DCMAKE_CXX_FLAGS=${SEAL_CXX_FLAGS}
   -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
   -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
   -DSEAL_USE_CXX17=${SEAL_USE_CXX17}
   # Skip updates
   UPDATE_COMMAND ""
)

ExternalProject_Get_Property(ext_seal SOURCE_DIR)
add_library(libseal STATIC IMPORTED)
set_target_properties(libseal PROPERTIES IMPORTED_LOCATION ${EXTERNAL_INSTALL_LIB_DIR}/libseal.a)
add_dependencies(libseal ext_seal)