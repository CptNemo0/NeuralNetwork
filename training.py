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

    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def __call__(self, sample, ):
        measured, target = sample['measured'], sample['targets']
        
        measured[0] = (measured[0] - self.x_min) / (self.x_max - self.x_min)
        measured[1] = (measured[1] - self.y_min) / (self.y_max - self.y_min)
        
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

        measured = self.coordinates.iloc[idx, :2]
        measured = np.array(measured)

        targets = self.coordinates.iloc[idx, 2:]
        targets = np.array(targets)
        
        sample = {'measured': measured, 'targets': targets}
 
        if self.transform:
           sample = self.transform(sample)

        return sample

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  
        self.fc1 = nn.Linear(2, 64)
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
    print(tdo)
    print(tdt)
    print()

def main():
    warnings.filterwarnings("ignore")
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    data = np.genfromtxt("dataset\\data.csv", delimiter=",")
    validation_data = np.genfromtxt("dataset\\validation_data.csv", delimiter=",")
    
    x_min = min(data[:,:1])
    y_min = min(data[:,1:2])
    x_max = max(data[:,:1])
    y_max = max(data[:,1:2])

    val_x_min = min(validation_data[:,:1])
    val_y_min = min(validation_data[:,1:2])
    val_x_max = max(validation_data[:,:1])
    val_y_max = max(validation_data[:,1:2])

    EPOCHS = 280
    learning_rate = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    net = Net()
    net.to(device)
    
    optimizer = optim.Adagrad(params = net.parameters(), lr = learning_rate)
    loss_function = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer, step_size = 50, gamma = 0.1)

    train_transform = transforms.Compose([Normalize(x_min, y_min, x_max, y_max), ToTensor()])
    validation_transform = transforms.Compose([Normalize(val_x_min, val_y_min, val_x_max, val_y_max), ToTensor()])

    training_set = CoordinatesCorrectionDataset("dataset\\data.csv", train_transform)
    validation_set = CoordinatesCorrectionDataset("dataset\\validation_data.csv", validation_transform)

    train_loader = torch.utils.data.DataLoader(training_set, batch_size = 32, shuffle = True, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = 8, shuffle = True, num_workers=4)

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
    
    modelPath = "model2\\model.pt"
    device = torch.device('cpu')
    model_scripted = torch.jit.script(net) 
    model_scripted.save(modelPath) 

if __name__ == "__main__":
    main()