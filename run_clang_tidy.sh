

#
python /usr/bin/run-clang-tidy-6.0.py \
-checks="-*,\
hicpp-*,\
-hicpp-no-array-decay,\
-hicpp-special-member-functions,\
-hicpp-vararg*,\
llvm-*,\
-llvm-include-order,\
misc-*,\
modernize-*,\
performance-*,\
"