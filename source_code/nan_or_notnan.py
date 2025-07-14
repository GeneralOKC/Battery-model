import numpy as np
import os
os.chdir("D:\\")

nan_arr = np.loadtxt('python\\p2d_fast_solver-main\\all_in\\nan.txt')
not_nan = np.loadtxt('python\\p2d_fast_solver-main\\all_in\\not_nan.txt')

print(np.shape(nan_arr), np.shape(not_nan))
print("matrix contains NaN:", np.isnan(nan_arr.data).any())
print("matrix contains Inf:", np.isinf(nan_arr.data).any())

print(np.linalg.det(nan_arr), np.linalg.det(not_nan))