
#!/bin/bash

# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
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

# list active docker containers
echo "Active docker containers..."
docker ps -a
echo

# clean up old docker containers
echo "Removing Exited docker containers..."
docker ps -a | grep Exited | cut -f 1 -d ' ' | xargs docker rm -f "${1}"
echo

#list docker images for he_transformer
echo "Docker images for he_transformer..."
docker images he_transformer
echo

# clean up docker images no longer in use
echo "Removing docker images for he_transformer..."
docker images -qa he_transformer* | xargs docker rmi -f "${1}"