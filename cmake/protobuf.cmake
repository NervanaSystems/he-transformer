# ******************************************************************************
# Copyright 2018-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# ******************************************************************************

include(ExternalProject)

set(PROTOBUF_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/protobuf)
set(PROTOBUF_SRC_DIR ${PROTOBUF_PREFIX}/src/ext_protobuf)
set(NGRAPH_HE_PROTOBUF_GIT_REPO_URL
    "https://github.com/protocolbuffers/protobuf")
set(NGRAPH_HE_PROTOBUF_GIT_TAG "v3.10.1")

message(STATUS "Installing protobuf to ${EXTERNAL_INSTALL_DIR}")

ExternalProject_Add(
  ext_protobuf
  PREFIX protobuf
  GIT_REPOSITORY ${NGRAPH_HE_PROTOBUF_GIT_REPO_URL}
  GIT_TAG ${NGRAPH_HE_PROTOBUF_GIT_TAG}
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  CONFIGURE_COMMAND ./autogen.sh
  COMMAND ./configure
          --prefix=${EXTERNAL_PROJECTS_ROOT}/protobuf
          --disable-shared
          CXX=${CMAKE_CXX_COMPILER}
  BUILD_COMMAND $(MAKE) "CXXFLAGS=-std=c++${NGRAPH_HE_CXX_STANDARD} -fPIC"
  TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/tmp"
  STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/stamp"
  DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/download"
  SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/src"
  BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf/src"
  INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/protobuf"
  EXCLUDE_FROM_ALL TRUE)

set(Protobuf_INSTALL_PREFIX ${EXTERNAL_PROJECTS_ROOT}/protobuf)
set(Protobuf_PROTOC_EXECUTABLE ${Protobuf_INSTALL_PREFIX}/bin/protoc)
set(Protobuf_INCLUDE_DIR ${Protobuf_INSTALL_PREFIX}/include)

set(Protobuf_LIBRARY ${Protobuf_INSTALL_PREFIX}/lib/libprotobuf.a)
set(Protobuf_LIBRARIES ${Protobuf_LIBRARY})

if(NOT TARGET protobuf::libprotobuf)
  add_library(protobuf::libprotobuf UNKNOWN IMPORTED)
  set_target_properties(protobuf::libprotobuf
                        PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                                   "${Protobuf_INCLUDE_DIR}"
                                   IMPORTED_LOCATION
                                   "${Protobuf_LIBRARY}")
  add_dependencies(protobuf::libprotobuf ext_protobuf)
endif()

if(NOT TARGET protobuf::protoc)
  add_executable(protobuf::protoc IMPORTED)
  set_target_properties(protobuf::protoc
                        PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                                   "${Protobuf_PROTOC_EXECUTABLE}"
                                   IMPORTED_LOCATION
                                   "${Protobuf_PROTOC_EXECUTABLE}")
  add_dependencies(protobuf::protoc ext_protobuf)
endif()

set(Protobuf_FOUND TRUE)
set(PROTOBUF_FOUND TRUE)

# Create copies of protobuf to allow pyhe_client to find it
add_custom_target(libprotobuf_soft_link ALL
                  DEPENDS libprotobuf
                  COMMAND ${CMAKE_COMMAND}
                          -E
                          make_directory
                          ${NGRAPH_TF_LIB_DIR}
                  COMMAND ${CMAKE_COMMAND}
                          -E
                          copy
                          ${EXTERNAL_INSTALL_LIB_DIR}/libprotobuf.so*
                          ${NGRAPH_TF_LIB_DIR})
