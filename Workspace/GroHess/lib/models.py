import numpy as np
import torch
import torch.nn as nn


#############################
#### Define vanilla ANN #####
#############################

class ANN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, random_seed):
        super().__init__()
        
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)

        # Ensure deterministic behavior in PyTorch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.fc1 = nn.Linear(num_inputs,num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        
    def forward(self, x) :
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x