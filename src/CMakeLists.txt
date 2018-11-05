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

# HE transformer sources
set (HE_SRC
    # kernels
    kernel/add.cpp
    kernel/broadcast.cpp
    kernel/convolution.cpp
    kernel/constant.cpp
    kernel/dot.cpp
    kernel/multiply.cpp
    kernel/negate.cpp
    kernel/pad.cpp
    kernel/reshape.cpp
    kernel/subtract.cpp
    kernel/result.cpp
    kernel/slice.cpp

    # seal
    seal/kernel/add_seal.cpp
    seal/kernel/multiply_seal.cpp
    seal/kernel/negate_seal.cpp
    seal/kernel/subtract_seal.cpp
    seal/he_seal_backend.cpp
    seal/seal_plaintext_wrapper.cpp
    seal/seal_ciphertext_wrapper.cpp
    seal/bfv/he_seal_bfv_backend.cpp
    seal/ckks/he_seal_ckks_backend.cpp

    # main
    he_backend.cpp
    he_cipher_tensor.cpp
    he_plain_tensor.cpp
    he_tensor.cpp
)

# Paths for the upstream ngraph
message(STATUS "NGRAPH_INCLUDE_DIR in parent scope: ${NGRAPH_INCLUDE_DIR}")
message(STATUS "NGRAPH_LIB_DIR in parent scope: ${NGRAPH_LIB_DIR}")

include_directories(
    SYSTEM
    ${NGRAPH_INCLUDE_DIR}
    ${HE_TRANSFORMER_SOURCE_DIR}
    ${EXTERNAL_INSTALL_INCLUDE_DIR}
)
link_directories(
    ${NGRAPH_LIB_DIR}
    ${EXTERNAL_INSTALL_LIB_DIR}
)

message("Linking directories: ${NGRAPH_LIB_DIR}, ${EXTERNAL_INSTALL_LIB_DIR}")

# Target library
add_library(he_backend SHARED ${HE_SRC})
add_dependencies(he_backend ext_ngraph ext_seal)
target_link_libraries(he_backend ngraph seal libjson)

set_target_properties(he_backend PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY
    ${EXTERNAL_NGRAPH_INSTALL_DIR}/lib
)

if (ENABLE_TF)
    # Install he-transformer to the python environment at `make install`
    execute_process(
        COMMAND python -c "import os; print(os.environ['VIRTUAL_ENV'])"
        OUTPUT_VARIABLE PYTHON_ENV
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    set (PY_NGRAPH_LIB_DIR ${PYTHON_ENV}/lib/python3.5/site-packages/ngraph)

    message("PY_NGRAPH_LIB_DIR ${PY_NGRAPH_LIB_DIR}")
    install(TARGETS he_backend DESTINATION ${PY_NGRAPH_LIB_DIR})
    install(DIRECTORY
        ${EXTERNAL_INSTALL_LIB_DIR}/
        DESTINATION ${PY_NGRAPH_LIB_DIR}
    )
endif()

# Install he-transformer (the HE libraries are installed separately) at `make install`
# By default, this installs to `~/ngraph_dist`
install(TARGETS he_backend DESTINATION ${NGRAPH_INSTALL_LIB_DIR})
install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION ${NGRAPH_INSTALL_INCLUDE_DIR}
    FILES_MATCHING PATTERN "*.hpp"
)
install(DIRECTORY
    ${EXTERNAL_INSTALL_LIB_DIR}/
    DESTINATION ${NGRAPH_INSTALL_LIB_DIR}
)
install(DIRECTORY
    ${EXTERNAL_INSTALL_INCLUDE_DIR}/
    DESTINATION ${NGRAPH_INSTALL_INCLUDE_DIR}
)
