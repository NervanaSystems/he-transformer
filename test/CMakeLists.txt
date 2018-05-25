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

# This dummy target always triggers re-configure of HE_HEAAN and HE_SEAL ngraph tests
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
    # test_basics_HE_SEAL.cpp
)
get_filename_component(MANIFEST he_seal_disabled_test.manifest ABSOLUTE)
add_custom_command(
    OUTPUT ${NGRAPH_TEST_HE_SEAL_SRC}
    DEPENDS ext_ngraph always_build
    COMMAND ${CMAKE_COMMAND}
            -DNGRAPH_TEST_DIR=${NGRAPH_TEST_DIR}
            -DBACKEND_NAME=HE_SEAL
            -DTEST_LOOPS=2
            -DMANIFEST=${MANIFEST}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/configure_test.cmake
)

# Configure test for HE_HEAAN
set (NGRAPH_TEST_HE_HEAAN_SRC
    backend_test_HE_HEAAN.cpp
    autodiff_HE_HEAAN.cpp
    convolution_test_HE_HEAAN.cpp
    # test_basics_HE_HEAAN.cpp
    )
get_filename_component(MANIFEST he_heaan_disabled_test.manifest ABSOLUTE)
add_custom_command(
    OUTPUT ${NGRAPH_TEST_HE_HEAAN_SRC}
    DEPENDS ext_ngraph always_build
    COMMAND ${CMAKE_COMMAND}
    -DNGRAPH_TEST_DIR=${NGRAPH_TEST_DIR}
    -DBACKEND_NAME=HE_HEAAN
    -DTEST_LOOPS=2
    -DMANIFEST=${MANIFEST}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/configure_test.cmake
)

# Build unit-test binary
set(SRC
    ${HE_TEST_SRC}
    ${NGRAPH_TEST_HE_SEAL_SRC}
    ${NGRAPH_TEST_HE_HEAAN_SRC})

#================================================================================================
# To auto generate a suite of unit tests for a backend add a line like this
# set(BACKEND_NAMES ${BACKEND_NAMES} "BACKEND_NAME_GOES_HERE")
# and replace BACKEND_NAME_GOES_HERE with your backend name.
# The code for the unit test suite is in test/backend_test.in.cpp
#================================================================================================
set(BACKEND_NAMES ${BACKEND_NAMES} "HE_SEAL", "HE_HEAAN")
message("FABI ${CMAKE_CURRENT_SOURCE_DIR}")

set(UNIT_TEST_CONFIG_LIST "" CACHE INTERNAL "")
foreach(TEST_CONFIG ${UNIT_TEST_CONFIG_LIST})
    string(FIND ${TEST_CONFIG} "@" OFFSET)
    string(SUBSTRING ${TEST_CONFIG} 0 ${OFFSET} BACKEND_NAME)
    math(EXPR OFFSET ${OFFSET}+1)
    string(SUBSTRING ${TEST_CONFIG} ${OFFSET} -1 MANIFEST)

    configure_file(backend_test.in.cpp backend_test_${BACKEND_NAME}.cpp)
    configure_file(convolution_test.in.cpp convolution_test_${BACKEND_NAME}.cpp)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_basics.in.cpp test_basics_${BACKEND_NAME}.cpp)
    set(SRC ${SRC} ${CMAKE_CURRENT_BINARY_DIR}/backend_test_${BACKEND_NAME}.cpp)
    set(SRC ${SRC} ${CMAKE_CURRENT_BINARY_DIR}/convolution_test_${BACKEND_NAME}.cpp)
    # set(SRC ${SRC} ${CMAKE_CURRENT_BINARY_DIR}/test_basics_${BACKEND_NAME}.cpp)

    # Some---but not all---autodiff tests go through multiple iterations with
    # different random seeds. On the CPU backend this is currently very slow
    # because the autodiff tests recompile with each iteration. That behavior
    # can be changed, but it's a bit involved, so for the time being we just
    # reduce the number of test iterations on non-INTERPRETER backends.
    if(${BACKEND_NAME} MATCHES ^INTERPRETER$)
        set(TEST_LOOPS 100)
    else()
        set(TEST_LOOPS 2)
    endif()

    configure_file(autodiff.in.cpp autodiff_${BACKEND_NAME}.cpp)
    set(SRC ${SRC} ${CMAKE_CURRENT_BINARY_DIR}/autodiff_${BACKEND_NAME}.cpp)

    message(STATUS "Adding unit test for backend ${BACKEND_NAME}")
endforeach()

add_executable(unit-test ${SRC})

target_link_libraries(unit-test he_backend seal heaan ntl ngraph_test_util libgtest pthread dl)

add_dependencies(unit-test he_backend ext_seal ext_heaan ext_ngraph ext_gtest)
