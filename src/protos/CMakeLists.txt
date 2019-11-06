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

message("PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}")

if(NOT Protobuf_FOUND)
  message(FATAL_ERROR "Cannot find Protobuf")
endif()

# Generate protobuf headers for message
get_filename_component(message_proto ${CMAKE_CURRENT_SOURCE_DIR}/message.proto
                       ABSOLUTE)
get_filename_component(message_proto_path "${message_proto}" PATH)
set(message_proto_srcs ${CMAKE_CURRENT_BINARY_DIR}/message.pb.cc PARENT_SCOPE)
set(message_proto_hdrs ${CMAKE_CURRENT_BINARY_DIR}/message.pb.h)

add_custom_command(OUTPUT ${message_proto_srcs} ${message_proto_hdrs}
                   COMMAND ${Protobuf_PROTOC_EXECUTABLE}
                           ARGS
                           --cpp_out
                           ${CMAKE_CURRENT_BINARY_DIR}
                           -I
                           ${message_proto_path}
                           ${message_proto}
                   DEPENDS ${message_proto} protobuf::protoc
                           protobuf::libprotobuf)

add_custom_target(generated_proto_files_target
                  DEPENDS ${message_proto_srcs} ${message_proto_hdrs})

# Create interface library; link against it to include the generated protobuf
# files
add_library(generated_proto_files INTERFACE)
target_link_libraries(generated_proto_files)
add_dependencies(generated_proto_files generated_proto_files_target)

# Use include protos/*.pb.h
target_include_directories(generated_proto_files
                           INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/../)

# Install generated protobuf files
foreach(INSTALL_INCLUDE_DIR ${INSTALL_INCLUDE_DIRS})
  message("Installing proto headers to ${INSTALL_INCLUDE_DIR}")
  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../
          DESTINATION ${INSTALL_INCLUDE_DIR}
          FILES_MATCHING
          PATTERN "*.hpp"
          PATTERN "*.h")
endforeach()

# Create dummy file with almost no clang-tidy checks. This prevents the
# generated protobuf files from being tidied. See https://
# gitlab.kitware.com/cmake/cmake/commit/b13bc8659f87567b1b091806d42f5023b2a6b48b
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/.clang-tidy" "
---
Checks: '-*,llvm-twine-local'
...
")
