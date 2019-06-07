# See here: https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
# for a description. We use mobilenet_v2_0.35_96.

mkdir -p model
cd model


# Get models
wget https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.35_96.tgz
tar -zxvf mobilenet_v2_0.35_96.tgz





cd -

transform_graph --in_graph=model/mobilenet_v2_0.35_96_frozen.pb \
                --out_graph=model/v2_035_96_opt.pb \
                --inputs="Placeholder" \
                --outputs="MobilenetV2/Logits/Conv2d_1c_1x1/BiasAdd" \
                --transforms='strip_unused_nodes
                              remove_nodes(op=Identity)
                              fold_constants(ignore_errors=true)
                              fold_batch_norms'

# For image processing
pip install pillow
