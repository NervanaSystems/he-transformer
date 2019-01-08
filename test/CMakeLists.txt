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

include_directories(SYSTEM
    ${GTEST_INCLUDE_DIR}
    ${NGRAPH_INCLUDE_DIR}
    ${NGRAPH_TEST_DIR}
    ${HE_TRANSFORMER_SOURCE_DIR}
    ${EXTERNAL_INSTALL_INCLUDE_DIR}
)
link_directories(
    ${NGRAPH_LIB_DIR}
    ${EXTERNAL_INSTALL_LIB_DIR}
)

set (SRC
    main.cpp
    test_seal.cpp
    test_util.cpp
)

set(ACTIVE_BACKEND_LIST HE_SEAL_BFV HE_SEAL_CKKS)

set(MULTI_TEST_SRC
    test_add.in.cpp
    test_avg_pool.in.cpp
    test_basics.in.cpp
    test_broadcast.in.cpp
    test_concat.in.cpp
    test_constant.in.cpp
    test_layers.in.cpp
    test_convolution.in.cpp
    test_dot.in.cpp
    test_multiply.in.cpp
    test_negate.in.cpp
    test_pad.in.cpp
    test_read_write.in.cpp
    test_reshape.in.cpp
    test_reverse.in.cpp
    test_slice.in.cpp
    test_subtract.in.cpp
    test_sum.in.cpp
)

foreach(BACKEND_NAME ${ACTIVE_BACKEND_LIST})
    string(REPLACE "HE_" "" BACKEND_DIR ${BACKEND_NAME})
    string(REPLACE "_" "/" BACKEND_DIR ${BACKEND_DIR})
    string(TOLOWER ${BACKEND_DIR} BACKEND_DIR)

    set(MANIFEST ${PROJECT_SOURCE_DIR}/src/${BACKEND_DIR}/unit_test.manifest)

    foreach(TEST_SRC ${MULTI_TEST_SRC})
        string(REPLACE ".in." "_${BACKEND_NAME}." TARGET_NAME ${TEST_SRC})
        configure_file(${TEST_SRC} ${TARGET_NAME})
        set(SRC ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME} ${SRC})
    endforeach()

    message(STATUS "Adding unit test for backend ${BACKEND_NAME}")
endforeach()

add_executable(unit-test ${SRC})

include_directories(SYSTEM
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HE_TRANSFORMER_SOURCE_DIR}
    ${EXTERNAL_INSTALL_INCLUDE_DIR}
    ${NGRAPH_TEST_UTIL_INCLUDE_DIR}
)

link_directories(
    ${NGRAPH_LIB_DIR}
    ${EXTERNAL_INSTALL_LIB_DIR}
    ${NGRAPH_TF_LIB_DIR}
)

target_include_directories(unit-test PRIVATE ".")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHE_SERIALIZED_ZOO=\\\"${CMAKE_CURRENT_SOURCE_DIR}/model\\\"")
if(NGRAPH_HE_SANITIZE_ADDRESS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer")
endif()

# Add ngraph test utils library
add_library(ngraph_test_util SHARED IMPORTED)
set_target_properties(
    ngraph_test_util
    PROPERTIES IMPORTED_LOCATION
    ${NGRAPH_TF_LIB_DIR}/libngraph_test_util.so
)

target_link_libraries(unit-test libgtest ngraph)
target_link_libraries(unit-test ngraph_test_util ${CMAKE_DL_LIBS})
target_link_libraries(unit-test he_backend seal pthread )

add_dependencies(unit-test he_backend ext_seal ext_ngraph_tf ext_gtest)

# Cryptonets
set (CRYPTONETS_SRC test_cryptonets.cpp test_util.cpp main_cryptonets.cpp)
add_executable(cryptonets_benchmark ${CRYPTONETS_SRC})

include_directories(SYSTEM
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HE_TRANSFORMER_SOURCE_DIR}
    ${EXTERNAL_INSTALL_INCLUDE_DIR}
    ${NGRAPH_TF_INCLUDE_DIR}
    ${NGRAPH_TEST_UTIL_INCLUDE_DIR}
)

message("Linking with NGRAPH_TF_LIB_DIR ${NGRAPH_TF_LIB_DIR}")

link_directories(
    ${NGRAPH_LIB_DIR}
    ${EXTERNAL_INSTALL_LIB_DIR}
    ${NGRAPH_TF_LIB_DIR}
)

target_include_directories(cryptonets_benchmark PRIVATE ".")

target_link_libraries(cryptonets_benchmark libgtest ngraph ngraph_test_util)
target_link_libraries(cryptonets_benchmark he_backend seal pthread ${CMAKE_DL_LIBS})
add_dependencies(cryptonets_benchmark he_backend ext_seal ext_ngraph_tf ext_gtest)
