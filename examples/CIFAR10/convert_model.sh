
# Turn from Keras to Tensorflow pb
python convert.py

# Remove unused nodes, fold constants, batch norms, etc.
transform_graph --in_graph=model/tf_model.pb \
                --out_graph=model/opt1.pb \
                --inputs="import/input_1" \
                --outputs="output/BiasAdd" \
                --transforms='strip_unused_nodes
                              remove_nodes(op=Identity)
                              remove_control_dependencies
                              fold_constants(ignore_errors=true)'
                           #   fold_batch_norms'