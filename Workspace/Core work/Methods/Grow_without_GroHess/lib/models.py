import numpy as np
import torch
import torch.nn as nn



######################
#### ANN to grow #####
######################



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

        # Store parameters
        self.num_inputs, self.num_hidden, self.num_outputs = num_inputs, num_hidden, num_outputs

        # Initialize the layers
        self.fc1s = nn.ModuleList([])
        self.fc2s = nn.ModuleList([])

        self.activation = torch.sigmoid ## nn.ReLU()
        
    def forward(self, x) :
        # LAYER 1
        out_fc1 = self.fc1s[0](x)
        for layer in self.fc1s[1:] :
            out_fc1 += layer(x)
        out_fc1 = self.activation(out_fc1)

        # LAYER 2
        out_fc2 = self.fc2s[0](out_fc1)
        for layer in self.fc2s[1:] :
            out_fc2 += layer(out_fc1)

        return out_fc2
    
    def add_neurons(self) :
        self.fc1s.append(nn.Linear(self.num_inputs,self.num_hidden))
        self.fc2s.append(nn.Linear(self.num_hidden,self.num_outputs))
    
    def freeze_neurons(self) :
        for layer in self.fc1s[:-1] :
            for param in layer.parameters() :
                param.requires_grad = False
        for layer in self.fc2s[:-1] :
            for param in layer.parameters() :
                param.requires_grad = False