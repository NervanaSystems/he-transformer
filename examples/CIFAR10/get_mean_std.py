import sys
import numpy as np

nums = [float(i.strip()) for i in sys.stdin]
print('nums', nums)
print('Count', len(nums))
print("Mean", np.mean(nums))
print("Std", np.std(nums))
print()