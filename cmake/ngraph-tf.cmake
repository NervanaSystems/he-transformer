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
ExternalProject_Add(
   ext_ngraph_tf DEPENDS install_tensorflow
   GIT_REPOSITORY https://github.com/NervanaSystems/ngraph-tf.git
   GIT_TAG v0.7.0-rc0
   PREFIX ${NGRAPH_TF_CMAKE_PREFIX}
   # CONFIGURE_COMMAND pip install -U tensorflow
   UPDATE_COMMAND ""
   INSTALL_COMMAND make install && pip install python/dist/ngraph-0.7.0-py2.py3-none-linux_x86_64.whl
   TEST_COMMAND python -c "import ngraph"
   #BUILD_ALWAYS 1
)
# ExternalProject_Get_Property(ext_ngraph_tf source_dir)