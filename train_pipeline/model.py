import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, input_size, hidden_dim=100, num_classes=6):
        super(NN, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc1 = nn.Linear(input_size, hidden_dim*2)
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc5 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.fc6 = nn.Linear(hidden_dim // 2, num_classes)
        self.init_weights()

    def forward(self, x):
        x = self.bn(x)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.bn2(x)
        x = F.elu(self.fc4(x))
        x = F.elu(self.fc5(x))
        x = self.bn3(x)
        return F.softmax(self.fc6(x))


    def init_weights(self):
        modules = self.modules()
        for i, m in enumerate(modules):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)


if __name__ == "__main__":
    model = NN(53, 100)
    print(model)