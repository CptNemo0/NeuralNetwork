import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import warnings

def main():
    validation_data = np.genfromtxt("dataset\\4x2acc\\validation_data_acc.csv", delimiter=",")

    mean = np.mean(validation_data[:,:5], axis=0), 
    std = np.std(validation_data[:,:5], axis=0), 
    mean2 = np.mean(validation_data[:,5:], axis=0), 
    std2 = np.std(validation_data[:,5:], axis=0)

    #-78, 161, 1119, 520, 692, 548.8539, 768.3435

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_array = np.array([-78, 161, 1119, 520, 692])
    input_array = (input_array - mean) / std
    input_tensor = torch.from_numpy(input_array)
    input_tensor = input_tensor.type(torch.float)
    input_tensor = input_tensor.to(device)

    model = torch.jit.load('model3\\model.pt')
    model.to(device)
    model.eval()

    output = model(input_tensor)
    output = output.to('cpu')

    output = output.detach().numpy()

    output = output * std2 + mean2

    print(output)


if __name__ == "__main__":
    main()