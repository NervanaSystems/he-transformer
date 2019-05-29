transform_graph --in_graph=model/tf_model.pb \
                --out_graph=model/opt1.pb \
                --inputs="Placeholder" \
                --outputs="MobilenetV2/Logits/Conv2d_1c_1x1/BiasAdd" \
                --transforms='strip_unused_nodes
                              remove_nodes(op=Identity)
                              fold_constants(ignore_errors=true)
                              fold_batch_norms'