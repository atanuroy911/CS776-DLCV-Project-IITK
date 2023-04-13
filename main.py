# Clear CACHE

# import torch
# torch.cuda.empty_cache()
#
# print(torch.cuda.memory_summary(device=None, abbreviated=False))

import torch

import sys
sys.path.insert(0, './yolov7')
# Load the PyTorch model
model = torch.load('./yolov7/weights/yolov7-tiny.pt')

# Extract model parameters
params = model.state_dict()

import numpy as np

for name, param in params.items():
    # Convert PyTorch tensor to NumPy array
    weight = param.cpu().numpy()

    # Save weight as .bin file
    bin_file = name + '.bin'
    weight.tofile(bin_file)

    # Save shape and type information as .param file
    param_info = np.array([weight.shape[0], weight.shape[1], param.dtype.itemsize], dtype=np.int32)
    param_file = name + '.param'
    param_info.tofile(param_file)
