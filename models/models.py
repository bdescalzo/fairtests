import torch.nn as nn
import torch.nn.functional as F


class GenericSmallModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x, params=None):
        if params is None:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = F.linear(x, params["fc1.weight"], params["fc1.bias"])
            x = F.relu(x)
            x = F.linear(x, params["fc2.weight"], params["fc2.bias"])
            x = F.relu(x)
            x = F.linear(x, params["fc3.weight"], params["fc3.bias"])
        return x.squeeze(-1)


class GenericModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x, params=None):
        if params is None:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = F.linear(x, params["fc1.weight"], params["fc1.bias"])
            x = F.relu(x)
            x = F.linear(x, params["fc2.weight"], params["fc2.bias"])
            x = F.relu(x)
            x = F.linear(x, params["fc3.weight"], params["fc3.bias"])
        if self.fc3.out_features == 1:
            return x.squeeze(-1)
        return x
