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

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        measured, target = sample['measured'], sample['targets']

        return {'measured': torch.from_numpy(measured),
                'targets': torch.from_numpy(target)}

class Normalize(object):
    """Normalize Input"""

    def __init__(self, mean, std, mean2, std2):
        self.mean = mean
        self.std = std
        self.mean2 = mean2
        self.std2 = std2

    def __call__(self, sample):
        measured, target = sample['measured'], sample['targets']
        
        measured = (measured - self.mean)/self.std
        target = (target - self.mean2)/self.std2
        
        return {'measured': measured,
                'targets': target}
    pass

class CoordinatesCorrectionDataset(Dataset):
    """Coordinates Correction dataset."""

    def __init__(self, csv_file, transform):
        """
        Arguments:
            csv_file (string): Path to the csv file with coordinates.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.coordinates = pd.read_csv(csv_file)
        self.transform = transform
        

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        measured = self.coordinates.iloc[idx, :5]
        measured = np.array(measured)

        targets = self.coordinates.iloc[idx, 5:]
        targets = np.array(targets)
        
        sample = {'measured': measured, 'targets': targets}
 
        if self.transform:
           sample = self.transform(sample)

        return sample

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  
        self.fc1 = nn.Linear(5,64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    
def train(model, epoch, trainLoader, device, optimizer, loss_function):
    model.train()

    print("Epoch: {}".format(epoch))

    for batch_idx, sample_batched in enumerate(trainLoader):
        measured, targets = sample_batched['measured'], sample_batched['targets']
        measured, targets = measured.type(torch.float), targets.type(torch.float),
        measured, targets = measured.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(measured)
        loss = loss_function(output, targets)
        loss.backward()
        optimizer.step()

def validation(model, testLoader, device, loss_funciton):
    model.eval()
    test_loss = 0
    correct = 0
    amount = len(testLoader.dataset)

    tdo : torch.tensor
    tdt : torch.tensor

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(testLoader):
            measured, targets = sample_batched['measured'], sample_batched['targets']
            measured, targets = measured.type(torch.float), targets.type(torch.float),
            measured, targets = measured.to(device), targets.to(device)
            output = model(measured)
            tdo = output
            tdt = targets
            loss = loss_funciton(output, targets)
            test_loss = test_loss + loss.item()
           
    test_loss = test_loss / amount

    print("\nValidation set:\n\tLoss: {:.5f}".format(test_loss))
    print("output")
    print(tdo[0])
    print("targets")
    print(tdt[0])
    print()

def main():
    warnings.filterwarnings("ignore")
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    data = np.genfromtxt("dataset\\4x2acc\\training_data_acc.csv", delimiter=",")
    validation_data = np.genfromtxt("dataset\\4x2acc\\validation_data_acc.csv", delimiter=",")

    EPOCHS = 280
    learning_rate = 0.15
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    net = Net()
    net.to(device)
    
    optimizer = optim.Adagrad(params = net.parameters(), lr = learning_rate, lr_decay = 0.001)
    loss_function = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, 0.9, 10)

    train_transform = transforms.Compose([
        Normalize(
                
                  mean = np.mean(data[:,:5], axis=0), 
                  std = np.std(data[:,:5], axis=0), 
                  mean2 = np.mean(data[:,5:], axis=0), 
                  std2 = np.std(data[:,5:], axis=0),
                ), 
        ToTensor()
        ])
    
    validation_transform = transforms.Compose([
        Normalize(
                mean = np.mean(validation_data[:,:5], axis=0), 
                std = np.std(validation_data[:,:5], axis=0), 
                mean2 = np.mean(validation_data[:,5:], axis=0), 
                std2 = np.std(validation_data[:,5:], axis=0)
                ), 
        ToTensor()
        ])

    training_set = CoordinatesCorrectionDataset("dataset\\4x2acc\\training_data_acc.csv", train_transform)
    validation_set = CoordinatesCorrectionDataset("dataset\\4x2acc\\validation_data_acc.csv", validation_transform)

    train_loader = torch.utils.data.DataLoader(training_set, batch_size = 32, shuffle = True, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = 32, shuffle = True, num_workers=4)
 
    modelPath = "model4\\model.pt"
    ets = []
    for epoch in range(1, EPOCHS + 1):
        start = time.perf_counter()

        train(net, epoch, train_loader, device, optimizer, loss_function)
        validation(net, validation_loader, device, loss_function)
        scheduler.step()

        stop = time.perf_counter()
        et = stop - start
        ets.append(et)
        output = "\tEpoch's time: {}, aprox time till the end: {}\n".format(et, (EPOCHS + 1 - epoch) * (sum(ets)/len(ets)))
        print(output)

        model_scripted = torch.jit.script(net) 
        model_scripted.save(modelPath) 
    
if __name__ == "__main__":
    main()