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

#configure_file(${NGRAPH_TEST_DIR}/backend_test.in.cpp backend_test_${BACKEND_NAME}.cpp)
# configure_file(${NGRAPH_TEST_DIR}/autodiff.in.cpp autodiff_${BACKEND_NAME}.cpp)
# configure_file(${NGRAPH_TEST_DIR}/convolution_test.in.cpp convolution_test_${BACKEND_NAME}.cpp)
# configure_file(${NGRAPH_TEST_DIR}/backend_test.in.cpp backend_test_${BACKEND_NAME}.cpp)
configure_file(${HE_TRANSFORMER_TEST_DIR}/test_basics.in.cpp test_basics_${BACKEND_NAME}${FUNCTION_NAME}.cpp)
configure_file(${HE_TRANSFORMER_TEST_DIR}/test_mnist.in.cpp test_mnist_${BACKEND_NAME}.cpp)
