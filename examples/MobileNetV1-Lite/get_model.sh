
cd model
wget http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_128.tgz

tar -zxvf mobilenet_v1_0.25_128.tgz

cd -


# transform model

# Maybe MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd instead
transform_graph --in_graph=model/mobilenet_v1_0.25_128_frozen.pb \
                --out_graph=model/opt1.pb \
                --inputs="Placeholder" \
                --outputs="MobilenetV1/Logits/SpatialSqueeze" \
                --transforms='strip_unused_nodes remove_nodes(op=Identity)'
