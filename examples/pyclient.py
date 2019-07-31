# ==============================================================================
#  Copyright 2018-2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================

import he_seal_client
import time

data = (1, 2, 3, 4)

hostname = 'localhost'
port = 34000
batch_size = 1

client = he_seal_client.HESealClient(hostname, port, batch_size, data, False)

while not client.is_done():
    time.sleep(1)

results = client.get_results()

print('results', results)
