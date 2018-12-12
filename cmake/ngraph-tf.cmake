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

include(ExternalProject)

set(EXTERNAL_NGRAPH_TF_INSTALL_DIR ${EXTERNAL_INSTALL_DIR})
set(NGRAPH_TF_CMAKE_PREFIX ext_ngraph_tf)

set(NGRAPH_TF_REPO_URL https://github.com/NervanaSystems/ngraph-tf.git)
set(NGRAPH_TF_GIT_LABEL v0.9.0)
set(COMPILE_FLAGS ${CMAKE_CXX_FLAGS})

message("Compile flags ${COMPILE_FLAGS}")

message("ng-tf CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER}")
message("ng-tf CMAKE_CXX_COMPILER ${CMAKE_C_COMPILER}")
message("ng-tf CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}")

message("ng-tf PY_NGRAPH_LIB_DIR ${PY_NGRAPH_LIB_DIR} OOOO")

ExternalProject_Add(
   ext_ngraph_tf DEPENDS ext_ngraph # he_backend
   GIT_REPOSITORY ${NGRAPH_TF_REPO_URL}
   GIT_TAG ${NGRAPH_TF_GIT_LABEL}
   PREFIX ${NGRAPH_TF_CMAKE_PREFIX}
   CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               #-DCMAKE_CXX_FLAGS=${COMPILE_FLAGS}
               -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_DIR}
               -DUSE_PRE_BUILT_NGRAPH=ON
               -DNGRAPH_ARTIFACTS_DIR=${EXTERNAL_INSTALL_DIR}

   # TODO: try below instead; note, make install ensures we copy he_backend.so to the CreatePipWhl.cmake first.
   INSTALL_COMMAND pip install ngraph-tensorflow-bridge
   # Work-around for ngraph-tf to recognize backends. TODO: modify ngraph-tf instead
   #INSTALL_COMMAND pip install python/dist/ngraph_tensorflow_bridge-0.8.0-py2.py3-none-manylinux1_x86_64.whl
   TEST_COMMAND python -c "import tensorflow as tf; print('TensorFlow version: r',tf.__version__);import ngraph_bridge; print(ngraph_bridge.__version__)"
   # BUILD_ALWAYS 1
)