
# True, True
echo 'True, True'
cat exp/cnn_bn_train_poly_exp_*.txt | grep Accuracy | cut -c11- | python get_mean_std.py

# True, False
echo 'True, False'
cat exp/cnn_bn_exp_*.txt | grep Accuracy | cut -c11- | python get_mean_std.py

# False, True
echo 'False, True'
cat exp/cnn_train_poly_exp_*.txt | grep Accuracy | cut -c11- | python get_mean_std.py

# False, False
echo 'False, False'
cat exp/cnn_exp_*.txt | grep Accuracy | cut -c11- | python get_mean_std.py