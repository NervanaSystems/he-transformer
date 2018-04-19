# ******************************************************************************
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
# ******************************************************************************

set (SRC
    gtest_main.cpp
    test_seal.cpp
)

include_directories(
    SYSTEM
    ${GTEST_INCLUDE_DIR}
    ${NGRAPH_HE_INSTALL_INCLUDE_DIR}
)
link_directories(${NGRAPH_HE_INSTALL_LIB_DIR})

add_executable(test-he ${SRC})
add_dependencies(test-he ext_gtest ext_seal)
set_target_properties(test-he PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXTERNAL_NGRAPH_TEST_DIR})
target_link_libraries(test-he libgtest seal pthread)
target_link_libraries(test-he ${CMAKE_DL_LIBS})
