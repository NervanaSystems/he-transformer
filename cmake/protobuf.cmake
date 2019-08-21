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
set(
  PROTOBUF_REPO_URL
  https://github.com/protocolbuffers/protobuf/releases/download/v3.9.1/protobuf-cpp-3.9.1.tar.gz
  )

ExternalProject_Add(
  ext_protobuf
  PREFIX protobuf
  URL ${PROTOBUF_REPO_URL}
  URL_HASH
    SHA256=29a1db3b9bebcf054c540f13400563120ff29fbdd849b2c7a097ffe9d3d508eb
  CONFIGURE_COMMAND ${PROTOBUF_SRC_DIR}/configure
                    --prefix=${EXTERNAL_INSTALL_DIR}
  INSTALL_COMMAND make install
                  # UPDATE_COMMAND ""
  EXCLUDE_FROM_ALL TRUE)

ExternalProject_Get_Property(ext_protobuf SOURCE_DIR)
ExternalProject_Get_Property(ext_protobuf BINARY_DIR)

message(STATUS "protoc SOURCE_DIR ${SOURCE_DIR}")
message(STATUS "protoc BINARY_DIR ${BINARY_DIR}")

add_library(libprotobuf_orig SHARED IMPORTED)
set_target_properties(libprotobuf_orig
                      PROPERTIES IMPORTED_LOCATION
                                 ${EXTERNAL_INSTALL_DIR}/lib/libprotobuf.so)
target_include_directories(libprotobuf_orig
                           INTERFACE ${EXTERNAL_INSTALL_DIR}/include)
add_dependencies(libprotobuf_orig ext_protobuf)

add_executable(protoc IMPORTED)
set_target_properties(protoc
                      PROPERTIES IMPORTED_LOCATION ${BINARY_DIR}/src/protoc)

set(_PROTOBUF_PROTOC ${BINARY_DIR}/src/protoc)

# Generate protobuf headers
get_filename_component(hw_proto
                       ${PROJECT_SOURCE_DIR}/src/protos/helloworld.proto
                       ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.pb.h")

message(STATUS "_PROTOBUF_PROTOC ${_PROTOBUF_PROTOC}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "hw_proto_path ${hw_proto_path}")
message(STATUS "hw_proto ${hw_proto}")

add_custom_command(OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}"
                   COMMAND ${_PROTOBUF_PROTOC}
                           ARGS
                           --cpp_out
                           "${CMAKE_CURRENT_BINARY_DIR}"
                           -I
                           "${hw_proto_path}"
                           "${hw_proto}"
                   DEPENDS "${hw_proto}" libprotobuf_orig)

add_custom_target(protobuf_files ALL DEPENDS ${hw_proto_srcs} ${hw_proto_hdrs})

add_library(libprotobuf INTERFACE)
# Include generated *.pb.h files
target_include_directories(libprotobuf
                           INTERFACE ${CMAKE_CURRENT_BINARY_DIR}
                                     ${EXTERNAL_INSTALL_DIR} ${SOURCE_DIR}/src)
target_link_libraries(libprotobuf INTERFACE libprotobuf_orig)
add_dependencies(libprotobuf libprotobuf_orig protobuf_files)
