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

cmake_minimum_required(VERSION 2.8)

include(ExternalProject)

ExternalProject_Add(ext_grpc
                    PREFIX grpc
                    GIT_REPOSITORY https://github.com/grpc/grpc.git
                    GIT_TAG v1.22.0
                    INSTALL_COMMAND ""
                    UPDATE_COMMAND "")
message("PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}")

ExternalProject_Get_Property(ext_grpc SOURCE_DIR)
ExternalProject_Get_Property(ext_grpc BINARY_DIR)
add_library(libgrpc STATIC IMPORTED)
message("grpc SOURCE_DIR ${SOURCE_DIR}")
message("grpc BINARY_DIR ${BINARY_DIR}")
set_target_properties(libgrpc
                      PROPERTIES IMPORTED_LOCATION ${BINARY_DIR}/libgrpc.a)
target_include_directories(libgrpc SYSTEM INTERFACE ${SOURCE_DIR}/include)
add_dependencies(libgrpc ext_grpc)

# Add libprotobuf
add_library(libprotobuf STATIC IMPORTED)
set_target_properties(
  libprotobuf
  PROPERTIES IMPORTED_LOCATION
             ${BINARY_DIR}/third_party/protobuf/libprotobuf.a
             INTERFACE_INCLUDE_DIRECTORIES
             ${SOURCE_DIR}/third_party/protobuf/src)
add_dependencies(libgrpc libprotobuf)

# Add cares
add_library(libcares STATIC IMPORTED)
set_target_properties(
  libcares
  PROPERTIES IMPORTED_LOCATION
             ${BINARY_DIR}/third_party/cares/cares/lib/libcares.a)
add_dependencies(libgrpc libcares)

message(STATUS "protibuf include dirs ${SOURCE_DIR}/third_party/protobuf/src")

# Add libgrpc++
add_library(libgrpc++ STATIC IMPORTED)
set_target_properties(libgrpc++
                      PROPERTIES IMPORTED_LOCATION ${BINARY_DIR}/libgrpc++.a)
add_dependencies(libgrpc libgrpc++)

# Add libgpr
add_library(libgpr STATIC IMPORTED)
set_target_properties(libgpr
                      PROPERTIES IMPORTED_LOCATION ${BINARY_DIR}/libgpr.a)
add_dependencies(libgrpc libgpr)

# Add libaddress_sorting
add_library(libaddress_sorting STATIC IMPORTED)
set_target_properties(libaddress_sorting
                      PROPERTIES IMPORTED_LOCATION
                                 ${BINARY_DIR}/libaddress_sorting.a)
add_dependencies(libgrpc libaddress_sorting)

# Add libgrpc++_unsecure
add_library(libgrpc++_unsecure STATIC IMPORTED)
set_target_properties(libgrpc++_unsecure
                      PROPERTIES IMPORTED_LOCATION
                                 ${BINARY_DIR}/libgrpc++_unsecure.a)
add_dependencies(libgrpc libgrpc++_unsecure)

# Add libgrpc++_reflection
add_library(libgrpc++_reflection STATIC IMPORTED)
set_target_properties(libgrpc++_reflection
                      PROPERTIES IMPORTED_LOCATION
                                 ${BINARY_DIR}/libgrpc++_reflection.a)
add_dependencies(libgrpc libgrpc++_reflection)

# Add protoc exectuable
add_executable(protoc IMPORTED)
set_target_properties(protoc
                      PROPERTIES IMPORTED_LOCATION
                                 ${BINARY_DIR}/third_party/protobuf/protoc)

# Generate protobuf headers
get_filename_component(hw_proto ${PROJECT_SOURCE_DIR}/protos/helloworld.proto
                       ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

set(_PROTOBUF_PROTOC ${BINARY_DIR}/third_party/protobuf/protoc)
set(_GRPC_CPP_PLUGIN_EXECUTABLE ${BINARY_DIR}/grpc_cpp_plugin)

# Generated sources
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.grpc.pb.h")

message(STATUS "_PROTOBUF_PROTOC ${_PROTOBUF_PROTOC}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "hw_proto_path ${hw_proto_path}")
message(STATUS "_GRPC_CPP_PLUGIN_EXECUTABLE ${_GRPC_CPP_PLUGIN_EXECUTABLE}")
message(STATUS "hw_proto ${hw_proto}")

add_custom_target(generate_protos)
add_custom_command(
  TARGET generate_protos
  # OUTPUT "${hw_proto_srcs}"
  # "${hw_proto_hdrs}"
  # "${hw_grpc_srcs}"
  # "${hw_grpc_hdrs}"
  COMMAND ${_PROTOBUF_PROTOC}
          ARGS
          --grpc_out
          "${CMAKE_CURRENT_BINARY_DIR}"
          --cpp_out
          "${CMAKE_CURRENT_BINARY_DIR}"
          -I
          "${hw_proto_path}"
          --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
          "${hw_proto}"
  DEPENDS "${hw_proto}" libprotobuf)

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

add_dependencies(libgrpc generate_protos)
