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

import pyhe_client
import time
import argparse


def main(FLAGS):
    data = (1, 2, 3, 4)

    port = 34000
    batch_size = 1

    client = pyhe_client.HESealClient(
        FLAGS.hostname, port, batch_size,
        {'client_parameter_name': ('encrypt', data)})

    results = client.get_results()

    print('results', results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hostname', type=str, default='localhost', help='Hostname of server')

    FLAGS, unparsed = parser.parse_known_args()

    print(FLAGS)
    main(FLAGS)
