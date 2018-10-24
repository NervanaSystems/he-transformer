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

configure_file(${HE_TRANSFORMER_TEST_DIR}/test_convolution.in.cpp test_convolution_${BACKEND_NAME}.cpp)
configure_file(${HE_TRANSFORMER_TEST_DIR}/test_read_write.in.cpp test_read_write_${BACKEND_NAME}.cpp)
configure_file(${HE_TRANSFORMER_TEST_DIR}/test_add.in.cpp test_add_${BACKEND_NAME}.cpp)
configure_file(${HE_TRANSFORMER_TEST_DIR}/test_multiply.in.cpp test_multiply_${BACKEND_NAME}.cpp)
configure_file(${HE_TRANSFORMER_TEST_DIR}/test_mnist.in.cpp test_mnist_${BACKEND_NAME}.cpp)
