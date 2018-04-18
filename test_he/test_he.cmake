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

message("$$$$$$$$$$ in test_he.cmake $$$$$$$$$$$")
# NOTE: This file is included by `ngraph-he/test/CMakeLists.txt` with CMake `include()`.
#       - We separate this file to minimize changes in the `ngraph` repo.
#       - So techcally this file belongs to the `ngraph` repo, but not the `he-transformer` repo.
#       - Variables defined in this file will affect `ngraph-he/test/CMakeLists.txt`.

set (TEST_HE_SRC
    ${HE_TRANSFORMER_TEST_HE_SOURCE_DIR}/gtest_main.cpp
    ${HE_TRANSFORMER_TEST_HE_SOURCE_DIR}/test_seal.cpp
)

add_executable(test-he ${TEST_HE_SRC})
target_link_libraries(test-he ngraph he libgtest pthread)
target_link_libraries(test-he ${CMAKE_DL_LIBS})
