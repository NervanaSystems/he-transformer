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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

include(ExternalProject)

set(EXTERNAL_NGRAPH_INSTALL_DIR ${EXTERNAL_INSTALL_DIR})
set(NGRAPH_TF_CMAKE_PREFIX ext_ngraph_tf)

SET(NGRAPH_TF_REPO_URL https://github.com/tensorflow/ngraph-bridge.git)
SET(NGRAPH_TF_GIT_LABEL v0.14.0)

SET(NGRAPH_TF_SRC_DIR ${CMAKE_BINARY_DIR}/${NGRAPH_TF_CMAKE_PREFIX}/src/${NGRAPH_TF_CMAKE_PREFIX})
SET(NGRAPH_TF_BUILD_DIR ${NGRAPH_TF_SRC_DIR}/build_cmake)
SET(NGRAPH_TF_ARTIFACTS_DIR ${NGRAPH_TF_BUILD_DIR}/artifacts)

SET(NGRAPH_TF_VENV_DIR ${NGRAPH_TF_BUILD_DIR}/venv-tf-py3)
SET(NGRAPH_TF_VENV_LIB_DIR ${NGRAPH_TF_VENV_DIR}/lib/${PYTHON_VENV_VERSION}/site-packages/ngraph_bridge)

SET(NGRAPH_TF_INCLUDE_DIR ${NGRAPH_TF_ARTIFACTS_DIR}/include)
SET(NGRAPH_TF_LIB_DIR ${NGRAPH_TF_ARTIFACTS_DIR}/lib)

SET(NGRAPH_TEST_UTIL_INCLUDE_DIR ${NGRAPH_TF_BUILD_DIR}/ngraph/test)

message("NGRAPH_TF_VENV_LIB_DIR ${NGRAPH_TF_VENV_LIB_DIR}")
message("NGRAPH_TF_LIB_DIR ${NGRAPH_TF_LIB_DIR}")

if (USE_PREBUILT_TF)
    message(STATUS "Using prebuilt TF")
    ExternalProject_Add(
        ext_ngraph_tf
        GIT_REPOSITORY ${NGRAPH_TF_REPO_URL}
        GIT_TAG ${NGRAPH_TF_GIT_LABEL}
        PREFIX ${NGRAPH_TF_CMAKE_PREFIX}
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_IN_SOURCE 1
        BUILD_BYPRODUCTS ${NGRAPH_TF_CMAKE_PREFIX}
        BUILD_COMMAND python3 ${NGRAPH_TF_SRC_DIR}/build_ngtf.py --use_prebuilt_tensorflow
        INSTALL_COMMAND ln -fs ${NGRAPH_TF_VENV_DIR} ${EXTERNAL_INSTALL_DIR}
)
else()
    message(STATUS "Rebuilding TF")
    ExternalProject_Add(
        ext_ngraph_tf
        GIT_REPOSITORY ${NGRAPH_TF_REPO_URL}
        GIT_TAG ${NGRAPH_TF_GIT_LABEL}
        PREFIX ${NGRAPH_TF_CMAKE_PREFIX}
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_IN_SOURCE 1
        BUILD_BYPRODUCTS ${NGRAPH_TF_CMAKE_PREFIX}
        BUILD_COMMAND python3 ${NGRAPH_TF_SRC_DIR}/build_ngtf.py
        INSTALL_COMMAND ln -fs ${NGRAPH_TF_VENV_DIR} ${EXTERNAL_INSTALL_DIR}
)
endif()

message("Copying files from ${NGRAPH_TF_LIB_DIR} to ${EXTERNAL_INSTALL_LIB_DIR}")
install(DIRECTORY ${NGRAPH_TF_LIB_DIR}/
        DESTINATION ${EXTERNAL_INSTALL_LIB_DIR}
        FILES_MATCHING
        PATTERN "*.so"
        PATTERN "*.so.*"
        PATTERN "*.a")

message("Copying TF files from ${NGRAPH_TF_INCLUDE_DIR} to ${EXTERNAL_INSTALL_INCLUDE_DIR}")
install(DIRECTORY ${NGRAPH_TF_INCLUDE_DIR}/
        DESTINATION ${EXTERNAL_INSTALL_INCLUDE_DIR}
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp")