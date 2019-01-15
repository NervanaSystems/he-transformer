import sys
import numpy as np

nums = [float(i.strip()) for i in sys.stdin]
print('nums', nums)
assert (len(nums) == 10)
#print('Count', len(nums))
print(
    np.round(np.mean(nums) * 100, 1),
    '$\pm$',
    np.round(np.std(nums) * 100, 1),
    end=' ')
