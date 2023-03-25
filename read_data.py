import numpy as np
import pickle

with open('train_label.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# 转换为 numpy 数组
loaded_data = np.array(loaded_data)
print(loaded_data)