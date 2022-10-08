import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self, input_size, hidden_dim, num_classes=6):
        super(NN, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        x = self.bn(x)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        return F.softmax(self.fc4(x))
