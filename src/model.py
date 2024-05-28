import torch
import torch.nn as nn
import torch.nn.functional as F


class model(nn.Module):
    def __init__(self, out_dim=10, in_channels=3, dataselection_object=None, classes=None):
        super().__init__()

        if dataselection_object:
            out_dim = dataselection_object.classcount
            in_channels = dataselection_object.channels
            fc1_size = dataselection_object.size_after_convolution

        if classes:
            self.classes_names = classes
        else:
            self.classes_names = {x: x for x in range(out_dim)}

        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(fc1_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
