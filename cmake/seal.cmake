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
SET(SEAL_GIT_TAG origin/3.2.0)

set(SEAL_USE_AES_NI_PRNG ON)
set(SEAL_USE_CXX17 ON)
set(SEAL_LIB_BUILD_TYPE "Static_PIC")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
   if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
      set(SEAL_USE_CXX17 OFF)
   endif()
endif()

message("CMAKE_CXX_COMPILER_ID ${CMAKE_CXX_COMPILER_ID}")

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
   set (SEAL_USE_AES_NI_PRNG OFF)
   set(SEAL_USE_CXX17 OFF)
   set(SEAL_LIB_BUILD_TYPE "Shared")
   message("SEAL_LIB_BUILD_TYPE ${SEAL_LIB_BUILD_TYPE}")
   message("SEAL_USE_AES_NI_PRNG ${SEAL_USE_AES_NI_PRNG}")
   message("SEAL_USE_CXX17 ${SEAL_USE_CXX17}")
endif()

message("SEAL_USE_CXX17 ${SEAL_USE_CXX17}")
message("INSTALL DIR ${EXTERNAL_INSTALL_DIR}")

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
   -DSEAL_USE_AES_NI_PRNG=${SEAL_USE_AES_NI_PRNG}
   -DSEAL_LIB_BUILD_TYPE=${SEAL_LIB_BUILD_TYPE}
)

add_custom_target(libseal ALL DEPENDS ext_seal ext_ngraph_tf
        COMMAND ${CMAKE_COMMAND} -E create_symlink
   ${EXTERNAL_INSTALL_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}seal${CMAKE_SHARED_LIBRARY_SUFFIX}
            ${NGRAPH_TF_VENV_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}seal${CMAKE_SHARED_LIBRARY_SUFFIX})
