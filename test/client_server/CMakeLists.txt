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
# WITHOUT WARRANTNNPS OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

# Client depends only on seal
include_directories(
    SYSTEM
    ${HE_TRANSFORMER_SOURCE_DIR}
    ${EXTERNAL_INSTALL_INCLUDE_DIR}
    ${NGRAPH_TF_INCLUDE_DIR}
)
message("HE_TRANSFORMER_SOURCE_DIR ${HE_TRANSFORMER_SOURCE_DIR}")
message("CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}")

add_library(he_seal_client SHARED ../../src/seal/he_seal_client.cpp)
add_dependencies(he_seal_client ext_seal)
target_link_libraries(he_seal_client seal)
install(TARGETS he_seal_client DESTINATION ${HE_INSTALL_LIB_DIR})

add_executable(main_client main_client.cpp)
add_dependencies(main_client ext_seal he_seal_client)
target_link_libraries(main_client seal he_seal_client)

# Server
add_executable(main_server main_server.cpp)
add_dependencies(main_server ext_ngraph_tf ext_seal)
target_link_libraries(main_server ngraph seal he_backend)
