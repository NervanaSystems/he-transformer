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
# WARRANTNNPS OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# ******************************************************************************

if(NOT Protobuf_FOUND)
  message(FATAL_ERROR "Cannot find Protobuf")
endif()

message(STATUS "pb CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "pb CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}")

# Generate protobuf headers for message
get_filename_component(message_proto ${CMAKE_CURRENT_SOURCE_DIR}/message.proto
                       ABSOLUTE)
get_filename_component(message_proto_path "${message_proto}" PATH)
set(message_proto_srcs ${CMAKE_CURRENT_BINARY_DIR}/message.pb.cc)
set(message_proto_hdrs ${CMAKE_CURRENT_BINARY_DIR}/message.pb.h)

add_custom_command(OUTPUT ${message_proto_srcs} ${message_proto_hdrs}
                   COMMAND ${Protobuf_PROTOC_EXECUTABLE}
                           ARGS
                           --cpp_out
                           ${CMAKE_CURRENT_SOURCE_DIR}
                           -I
                           ${message_proto_path}
                           ${message_proto}
                   DEPENDS ${message_proto} protobuf::protoc
                           protobuf::libprotobuf)

# add_custom_target(protobuf_files ALL DEPENDS ${message_proto_srcs}
# ${message_proto_hdrs})
#
# add_library(libprotobuf INTERFACE) Include generated *.pb.h files
# message(STATUS "Including pb directory ${CMAKE_CURRENT_BINARY_DIR}")
# target_include_directories(libprotobuf INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
# target_link_libraries(libprotobuf INTERFACE libprotobuf_orig)
# add_dependencies(libprotobuf libprotobuf_orig protobuf_files)
