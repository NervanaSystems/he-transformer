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

set(SRC
    main.cpp
    # src/
    test_he_op_annotations.cpp
    test_he_plaintext.cpp
    test_he_tensor.cpp
    test_he_type.cpp
    test_he_util.cpp
    test_node_wrapper.cpp
    # src/pass
    test_he_fusion.cpp
    test_he_supported_ops.cpp
    test_propagate_he_annotations.cpp
    # src/seal
    test_encryption_parameters.cpp
    test_he_seal_executable.cpp
    test_bounded_relu.cpp
    test_perf_micro.cpp
    test_seal.cpp
    test_protobuf.cpp
    test_seal_util.cpp
    # src/tcp
    test_tcp_message.cpp
    test_tcp_client.cpp
    # test logging
    test_ngraph_he_log.cpp
    )

set(BACKEND_TEST_SRC
    test_add.in.cpp
    test_avg_pool.in.cpp
    test_batch_norm_inference.in.cpp
    test_basics.in.cpp
    test_broadcast.in.cpp
    test_concat.in.cpp
    test_constant.in.cpp
    test_convolution.in.cpp
    test_divide.in.cpp
    test_exp.in.cpp
    test_dot.in.cpp
    test_rescale.in.cpp
    test_layers.in.cpp
    test_maxpool.in.cpp
    test_minimum.in.cpp
    test_multiply.in.cpp
    test_max.in.cpp
    test_negate.in.cpp
    test_pad.in.cpp
    test_power.in.cpp
    test_read_write.in.cpp
    test_relu.in.cpp
    test_reshape.in.cpp
    test_reverse.in.cpp
    test_server_client.in.cpp
    test_softmax.in.cpp
    test_slice.in.cpp
    test_subtract.in.cpp
    test_sum.in.cpp)

set(MANIFEST ${PROJECT_SOURCE_DIR}/src/seal/unit_test.manifest)
set(BACKEND_NAME "HE_SEAL")

foreach(TEST_SRC ${BACKEND_TEST_SRC})
  string(REPLACE ".in."
                 "_${BACKEND_NAME}."
                 TARGET_NAME
                 ${TEST_SRC})
  configure_file(${TEST_SRC} ${TARGET_NAME})
  set(SRC ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME} ${SRC})
endforeach()

message(STATUS "Adding unit test for backend ${BACKEND_NAME}")

add_executable(unit-test ${SRC})

target_include_directories(unit-test PRIVATE ".")

# Add ngraph test utils library
add_library(ngraph_test_util STATIC IMPORTED)
set_target_properties(
  ngraph_test_util
  PROPERTIES
    IMPORTED_LOCATION
    ${NGRAPH_TF_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ngraph_test_util${CMAKE_STATIC_LIBRARY_SUFFIX}
  )
add_dependencies(ngraph_test_util ext_ngraph_tf)
# Set the test_utils include folder manually. The
# ${NGRAPH_TF_BUILD_DIR}/ngraph/test/ folder doesn't exist until ngraph-tf
# builds, so we cannot set target_include_directories(ngraph_test_util SYSTEM
# INTERFACE ${NGRAPH_TF_BUILD_DIR}/ngraph/test/) (Note, the work-around
# file(MAKE_DIRECTORY ${NGRAPH_TF_BUILD_DIR}/ngraph/test/)) doesn't work either,
# as ngraph-tf will complain about the ngraph folder already existing.
target_include_directories(unit-test
                           PRIVATE ${NGRAPH_TF_BUILD_DIR}/ngraph/test/)

target_link_libraries(unit-test PRIVATE ngraph_test_util)
target_link_libraries(unit-test PRIVATE ngraph libgtest)
target_link_libraries(unit-test PRIVATE he_seal_backend libseal)
target_link_libraries(unit-test PRIVATE protobuf::libprotobuf)

add_custom_target(check
                  COMMAND ${PROJECT_BINARY_DIR}/test/unit-test \${ARGS}
                  DEPENDS unit-test)
