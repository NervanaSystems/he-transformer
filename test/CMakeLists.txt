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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHE_SERIALIZED_ZOO=\\\"${CMAKE_CURRENT_SOURCE_DIR}/model\\\"")

set (HE_TEST_SRC
    main.cpp
    #test_basics.cpp
    test_heaan.cpp
    #test_model.cpp
    #test_mnist.cpp
    #test_noise.cpp
    #test_overflow.cpp
    test_seal.cpp
    #test_speed.cpp
    test_util.cpp
)

# This dummy target always triggers re-configure of NNP and NNPI ngraph tests
# If the source file is not changed, re-compile is not triggered
add_custom_command(
    OUTPUT always_build
    COMMAND cmake -E echo
)

# Configure test for HE_SEAL
set (NGRAPH_TEST_HE_SEAL_SRC
    backend_test_HE_SEAL.cpp
    autodiff_HE_SEAL.cpp
    convolution_test_HE_SEAL.cpp
)
get_filename_component(MANIFEST he_seal_disabled_test.manifest ABSOLUTE)
add_custom_command(
    OUTPUT ${NGRAPH_TEST_HE_SEAL_SRC}
    DEPENDS ext_ngraph always_build
    COMMAND ${CMAKE_COMMAND}
            -DNGRAPH_TEST_DIR=${NGRAPH_TEST_DIR}
            -DBACKEND_NAME=HE_SEAL
            -DTEST_LOOPS=200
            -DMANIFEST=${MANIFEST}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/configure_test.cmake
)

# Configure test for HE_HEAAN
set (NGRAPH_TEST_HE_HEAAN_SRC
    backend_test_HE_HEAAN.cpp
    autodiff_HE_HEAAN.cpp
    convolution_test_HE_HEAAN.cpp
    )
get_filename_component(MANIFEST he_heaan_disabled_test.manifest ABSOLUTE)
add_custom_command(
    OUTPUT ${NGRAPH_TEST_HE_HEAAN_SRC}
    DEPENDS ext_ngraph always_build
    COMMAND ${CMAKE_COMMAND}
    -DNGRAPH_TEST_DIR=${NGRAPH_TEST_DIR}
    -DBACKEND_NAME=HE_HEAAN
    -DTEST_LOOPS=200
    -DMANIFEST=${MANIFEST}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/configure_test.cmake
)

# Build unit-test binary
set(SRC
    ${HE_TEST_SRC}
    ${NGRAPH_TEST_HE_SEAL_SRC}
    ${NGRAPH_TEST_HE_HEAAN_SRC})
add_executable(unit-test ${SRC})

target_link_libraries(unit-test he_backend seal heaan ntl ngraph_test_util libgtest pthread dl)

add_dependencies(unit-test he_backend ext_seal ext_heaan ext_ngraph ext_gtest)
