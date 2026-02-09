import torch
import numpy as np
tensor_2d = torch.randn(3, 4)
array = np.random.randn(3, 4)
## print("Tensor-2D:\n", tensor_2d)
## print("Array:\n", array)
tensor_3d = torch.randn(3, 2, 4)
## print("Tensor-3D:\n", tensor_3d)
## create tensor from array
tensor_from_array = torch.tensor(array)
## print("Tensor from Array:\n", tensor_from_array)
