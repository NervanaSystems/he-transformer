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
# *****************************************************************************

import os
import sys


# See here: https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
# for a description of models used
def main():
    if not os.path.exists('model'):
        os.mkdir('model')
    os.chdir(os.path.join(sys.path[0], 'model'))

    url_prefix = 'https://storage.googleapis.com/mobilenet_v2/checkpoints/'
    filename_suffix = '.tgz'
    filename_prefixes = [
        'mobilenet_v2_0.35_96', 'mobilenet_v2_1.0_96', 'mobilenet_v2_1.4_224',
        'mobilenet_v2_0.35_128', 'mobilenet_v2_0.35_160',
        'mobilenet_v2_0.5_96', 'mobilenet_v2_0.5_128', 'mobilenet_v2_0.75_96',
        'mobilenet_v2_0.75_128'
    ]
    pb_suffix = '_frozen.pb'
    opt_suffix = '_opt.pb'

    for filename_prefix in filename_prefixes:
        filename = filename_prefix + filename_suffix
        if not os.path.isfile(filename):
            os.system('wget ' + url_prefix + filename)

        pb_filename = filename_prefix + pb_suffix

        if not os.path.isfile(pb_filename):
            os.system('tar -zxvf ' + filename)

        opt_filename = filename_prefix + opt_suffix

        if not os.path.isfile(opt_filename):
            transform_cmd = 'transform_graph --in_graph='
            transform_cmd += pb_filename
            transform_cmd += ' --out_graph='
            transform_cmd += opt_filename
            transform_cmd += ' --inputs="Placeholder"'
            transform_cmd += ' --outputs="MobilenetV2/Logits/Conv2d_1c_1x1/BiasAdd"'
            transform_cmd += ''' --transforms="strip_unused_nodes remove_nodes(op=Identity)'''
            transform_cmd += ''' fold_constants(ignore_errors=true) fold_batch_norms"'''
            print('transform_cmd')
            print(transform_cmd)
            os.system(transform_cmd)


if __name__ == '__main__':
    main()