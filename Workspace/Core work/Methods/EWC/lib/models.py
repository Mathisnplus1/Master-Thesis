import numpy as np
import torch
import torch.nn as nn


##############
#### ANN #####
##############


class ANN (nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, random_seed):
        super().__init__()
        
        # Reproducibility
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Initialize the layers
        self.fc1 = nn.Linear(num_inputs,num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_outputs)

        self.activation = torch.sigmoid ## nn.ReLU()
        
    def forward(self, x) :
        x = x.view(-1, 28*28)
        # LAYER 1
        out_fc1 = self.fc1(x)
        out_fc1 = self.activation(out_fc1)

        # LAYER 2
        out_fc2 = self.fc2(out_fc1)
        out_fc2 = self.activation(out_fc2)

        # LAYER 3
        out_fc3 = self.fc3(out_fc2)

        return out_fc3