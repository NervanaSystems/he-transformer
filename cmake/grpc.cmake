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
                    INSTALL_COMMAND "")

ExternalProject_Get_Property(ext_grpc SOURCE_DIR)
ExternalProject_Get_Property(ext_grpc BINARY_DIR)
add_library(libgrpc STATIC IMPORTED)
message("grpc SOURCE_DIR ${SOURCE_DIR}")
message("grpc BINARY_DIR ${BINARY_DIR}")
set_target_properties(libgrpc
                      PROPERTIES IMPORTED_LOCATION ${BINARY_DIR}/libgrpc.a)

target_include_directories(libgrpc SYSTEM INTERFACE ${SOURCE_DIR}/include)
add_dependencies(libgrpc ext_grpc)
