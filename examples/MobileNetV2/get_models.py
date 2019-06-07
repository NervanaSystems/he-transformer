import os

# See here: https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
# for a description. We use mobilenet_v2_0.35_96.


def main():

    os.system('mkdir -p model && cd model')

    url_prefix = 'https://storage.googleapis.com/mobilenet_v2/checkpoints/'
    filename_suffix = '.tgz'
    filename_prefixes = ['mobilenet_v2_0.35_96', 'mobilenet_v2_1.0_96']
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

#main()
# Get models
#
#tar -zxvf mobilenet_v2_0.35_96.tgz

#cd -

# For image processing
#pip install pillow
