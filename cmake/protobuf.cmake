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

set(NGRAPH_HE_PROTOBUF_GIT_REPO_URL
    "https://github.com/protocolbuffers/protobuf")
set(NGRAPH_HE_PROTOBUF_GIT_TAG "v3.10.1")

message("MAKE $(MAKE)")
message("NGRAPH_HE_CXX_STANDARD ${NGRAPH_HE_CXX_STANDARD}")

ExternalProject_Add(
  ext_protobuf
  PREFIX protobuf
  GIT_REPOSITORY ${NGRAPH_HE_PROTOBUF_GIT_REPO_URL}
  GIT_TAG ${NGRAPH_HE_PROTOBUF_GIT_TAG}
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  CONFIGURE_COMMAND ./autogen.sh COMMAND ./configure --prefix=${EXTERNAL_PROJECTS_ROOT}/protobuf --disable-shared CXX=${CMAKE_CXX_COMPILER} CC=${CMAKE_C_COMPILER}
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
  add_library(protobuf::libprotobuf STATIC IMPORTED)
  if(NOT EXISTS ${Protobuf_INCLUDE_DIR})
    file(MAKE_DIRECTORY ${Protobuf_INCLUDE_DIR})
  endif()

  target_include_directories(protobuf::libprotobuf SYSTEM
                             INTERFACE "${Protobuf_INCLUDE_DIR}")
  set_target_properties(protobuf::libprotobuf
                        PROPERTIES IMPORTED_LOCATION "${Protobuf_LIBRARY}")
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
