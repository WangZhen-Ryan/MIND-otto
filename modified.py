# 1,6,7,8,9
import pandas as pd
import numpy as np
train_path = "train.txt"
val_path = "test.txt"

read_file = pd.read_csv(r'train.txt',delimiter='\t',header=None)
read_file = read_file.iloc[:,[0,5,6,7,8]]

np.savetxt("train_modified.txt", read_file.values, fmt='%s',delimiter='\t')


read_file = pd.read_csv(r'test.txt',delimiter='\t',header=None)
read_file = read_file.iloc[:,[0,5,6,7,8]]

np.savetxt("test_modified.txt", read_file.values, fmt='%s',delimiter='\t')