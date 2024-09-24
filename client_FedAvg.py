from collections import OrderedDict

import os
import math
import torch
import typer
from PIL import Image
import torch.nn as nn
from torch import optim
from logging import INFO, DEBUG
import torch.nn.functional as F
from skimage import io, transform
from flwr.common.logger import log
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import models
import flwr as fl


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Define data transformations
#data_transforms = { ....
#}


class FL_LUS_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.X = []
        self.y = []
        self.transform = data_transforms
        fold = root_dir.split("_")[-1]
        
        # Read data
        for score in range(4): #4 due to 4 scores
            subdir = os.path.join(root_dir, f"Score{score}")
            for file in os.listdir(subdir):
                image = Image.open(os.path.join(subdir, file))
                image = self.transform[fold](image)
                self.X.append(image)
                self.y.append(score)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


#Define the model as it was defined on the server side
model = get_model_R18_SA(num_classes=4).to(device)
def load_data(train_dir, center_name):
    class_weights = []
    for i in range(4): #4 due to 4 scores
        nfiles = len(os.listdir(f"{train_dir}/{center_name}_Train/Score{i}"))
        if nfiles > 0:
            class_weights.append(1/nfiles)
        else:
            class_weights.append(0)
            
    data_dir = " "
    image_datasets = {x: FL_LUS_Dataset(os.path.join(data_dir, f"{center_name}_{x}"))
                      for x in ['Train', 'Test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Test']}


    trainloader = torch.utils.data.DataLoader(image_datasets['Train'], batch_size=4,
                                                 shuffle=True, num_workers=4)

    testloader = torch.utils.data.DataLoader(image_datasets['Test'], batch_size=4,
                                                 shuffle=False, num_workers=4)
    return trainloader, testloader, dataset_sizes, class_weights


def train(net, trainloader, epochs, class_weights):
    """Train the network on the training set."""
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights)).to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=0.0001, weight_decay = 0.00001)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader, class_weights):
    """Validate the network on the entire test set."""
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


class LUSClient(fl.client.NumPyClient):
    def __init__(self, center_name):
        self._center_name = center_name
        self._trainloader, self._testloader, self._dataset_sizes, self._class_weights = load_data(data_dir, center_name)         
        self.net = mymodel().to(DEVICE)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self._trainloader, epochs=1, class_weights = self._class_weights)
        return self.get_parameters(config={}), len(self._trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self._testloader, class_weights = self._class_weights)
        log(INFO, f"{self._center_name}: {loss} {accuracy}")
        return float(loss), self._dataset_sizes["Test"], {"accuracy": float(accuracy), "loss": float(loss)}


def main(center_name):
    fl.client.start_numpy_client(server_address="[::]:8080", client=LUSClient(center_name))

if __name__ == "__main__":
    typer.run(main)
