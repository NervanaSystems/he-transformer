echo 'CF deeper experiment results'

echo '\hline'
# True, True
echo '\\centering{\\cmark} & \\centering{\\cmark} & \\centering{11} &'
cat deeper_exp/cnn_bn_train_poly_exp_*.txt | grep Accuracy | cut -c11- | python get_mean_std.py
echo '& \\textbf{TODO} \\\\'

# True, False
echo '\\centering{\\cmark} & \\centering{\\xmark} & \\centering{11} &'
cat deeper_exp/cnn_bn_exp_*.txt | grep Accuracy | cut -c11- | python get_mean_std.py
echo '& \\textbf{TODO} \\\\'

# False, True
echo '\\centering{\\xmark} & \\centering{\\cmark} & \\centering{11} &'
cat deeper_exp/cnn_train_poly_exp_*.txt | grep Accuracy | cut -c11- | python get_mean_std.py
echo '& \\textbf{TODO} \\\\'

# False, False
echo '\\centering{\\xmark} & \\centering{\\xmark} & \\centering{11} &'
cat deeper_exp/cnn_exp_*.txt | grep Accuracy | cut -c11- | python get_mean_std.py
echo '& \\textbf{TODO} \\\\'

echo '\hline'