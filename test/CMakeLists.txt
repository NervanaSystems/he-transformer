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

set (HE_TEST_SRC
    main.cpp
    test_basics.cpp
    test_heaan.cpp
    # test_model.cpp // TODO: re-enable
    test_mnist.cpp
    test_noise.cpp
    test_overflow.cpp
    test_seal.cpp
    test_speed.cpp
    test_util.cpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHE_SERIALIZED_ZOO=\\\"${CMAKE_CURRENT_SOURCE_DIR}/model\\\"")

# Configure test for HE
set (NGRAPH_TEST_SRC
    backend_test_HE.cpp
    autodiff_HE.cpp
    convolution_test_HE.cpp
)
add_custom_command(
    OUTPUT ${NGRAPH_TEST_SRC}
    DEPENDS ext_ngraph
    COMMAND ${CMAKE_COMMAND}
            -DNGRAPH_TEST_DIR=${NGRAPH_TEST_DIR}
            -DBACKEND_NAME=HE
            -DTEST_LOOPS=2
            -P ${CMAKE_CURRENT_SOURCE_DIR}/configure_test.cmake
)

set(SRC ${HE_TEST_SRC} ${NGRAPH_TEST_SRC})
add_executable(unit-test ${SRC})

target_link_libraries(unit-test he_backend seal heaan ntl ngraph_test_util libgtest pthread)

add_dependencies(unit-test he_backend ext_seal ext_heaan ext_ngraph ext_gtest)
