import torch
from torch import nn
from torch.nn import Module
from torch.nn import functional as F

class Model(Module):

    def __init__(self, n_input:int, n_output:int):
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output

        # Parameter of the model used for act and target on the algorithm
        nNeurons_1 = 250
        nNeurons_2 = 125
        nNeurons_3 = 60

        self.FC1 = nn.Linear(in_features=n_input, out_features=nNeurons_1)
        self.FC2 = nn.Linear(in_features=nNeurons_1, out_features=nNeurons_2)
        self.FC3 = nn.Linear(in_features=nNeurons_2, out_features=nNeurons_3)
        self.out = nn.Linear(in_features=nNeurons_3, out_features=n_output)

        # Initializing weights
        self.FC1.weight.data.normal_(std=0.1)
        self.FC2.weight.data.normal_(std=0.1)
        self.FC3.weight.data.normal_(std=0.1)
        self.out.weight.data.normal_(std=0.1)
    
    def forward(self, x):
        # declare the forward behavior here
        x = F.relu(self.FC1(x))
        x = F.relu(self.FC2(x))
        x = F.relu(self.FC3(x))
        x = self.out(x)
        return x