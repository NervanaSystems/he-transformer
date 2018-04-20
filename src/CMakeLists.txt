# *****************************************************************************
# Copyright 2017-2018 Intel Corporation
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
# *****************************************************************************

# HE transformer headers
set(HE_TRANSFORMER_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# HE transformer sources
set (HE_SRC
    he_backend.cpp
    he_call_frame.cpp
    he_external_function.cpp
    he_tensor_view.cpp
    he_cipher_tensor_view.cpp
    he_manager.cpp
)

# Paths for the upstream ngraph
message(STATUS "EXTERNAL_NGRAPH_INCLUDE_DIR in parent scope: ${EXTERNAL_NGRAPH_INCLUDE_DIR}")
message(STATUS "EXTERNAL_NGRAPH_LIB_DIR in parent scope: ${EXTERNAL_NGRAPH_LIB_DIR}")

include_directories(
    SYSTEM
    ${EXTERNAL_NGRAPH_INCLUDE_DIR}
    ${HE_TRANSFORMER_INCLUDE_DIR}
    ${NGRAPH_HE_INSTALL_INCLUDE_DIR}
)

link_directories(${NGRAPH_HE_INSTALL_LIB_DIR})

# Target library
add_library(he SHARED ${HE_SRC})
install(TARGETS he DESTINATION ${NGRAPH_HE_INSTALL_LIB_DIR})

add_dependencies(he ext_heaan ext_seal)
target_link_libraries(he seal)

# Installs header make `make install`
install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION "${NGRAPH_INSTALL_INCLUDE_DIR}"
    FILES_MATCHING PATTERN "*.hpp"
)
