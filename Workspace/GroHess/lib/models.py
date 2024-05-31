import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import svds



######################
#### Vanilla ANN #####
######################

class ANN_old(nn.Module):
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
        self.fc3 = nn.Linear(num_hidden, num_outputs)
        
    def forward(self, x) :
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
##################################
#### Function to add neurons #####
##################################

# Define the function to add neurons to a layer. It will be call in the ANN class
def add_neurons (fc1, fc2, fc1_weight_grad, fc1_bias_grad, fc2_weight_grad, num_neurons, device, init_name, growth_matrix, c) :
    if init_name == "random" :
        # PRE-LAYER

        num_in_1, num_out_1 = fc1.in_features, fc1.out_features
        new_fc1 = nn.Linear(in_features=num_in_1, out_features=num_out_1 + num_neurons).to(device)
        # Set parameters
        new_fc1.weight = nn.Parameter(torch.cat((fc1.weight,torch.randn(num_neurons, num_in_1).to(device))))
        new_fc1.bias = nn.Parameter(torch.cat((fc1.bias, torch.randn(num_neurons).to(device))))
        # Set gradients
        new_fc1.weight.grad = nn.Parameter(torch.cat((fc1_weight_grad,torch.randn(num_neurons, num_in_1).to(device)), dim=0))
        new_fc1.bias.grad = nn.Parameter(torch.cat((fc1_bias_grad, torch.randn(num_neurons).to(device)), dim=0))

        # POST-LAYER

        num_in_2, num_out_2 = fc2.in_features, fc2.out_features
        new_fc2 = nn.Linear(in_features=num_in_2 + num_neurons, out_features=num_out_2).to(device)
        # Set parameters
        new_fc2.weight = nn.Parameter(torch.cat((fc2.weight, torch.randn(num_out_2,num_neurons).to(device)), dim=1))
        new_fc2.bias = fc2.bias
        # Set gradients
        new_fc2.weight.grad = torch.cat((fc2_weight_grad, torch.randn(num_out_2, num_neurons).to(device)), dim=1)
        new_fc2.bias.grad = fc2.bias.grad

    elif init_name == "gradmax" :
        # Perform SVD
        u, s, _ = svds(growth_matrix.cpu().detach().numpy(), k=num_neurons, return_singular_vectors=True)
        eigenvals, eigenvecs = (s**2), u[::-1]
        scaler = c / np.sqrt(eigenvals.sum())
        scaler = 1
        added_weights_post_layer = (1/scaler)*torch.tensor(eigenvecs[:, :num_neurons].copy()).to(device)
        
        # PRE-LAYER

        num_in_1, num_out_1 = fc1.in_features, fc1.out_features
        new_fc1 = nn.Linear(in_features=num_in_1, out_features=num_out_1 + num_neurons).to(device)
        # Set parameters
        new_fc1.weight = nn.Parameter(torch.cat((fc1.weight,torch.zeros(num_neurons, num_in_1).to(device))))
        new_fc1.bias = nn.Parameter(torch.cat((fc1.bias, torch.zeros(num_neurons).to(device))))
        # Set gradients
        new_fc1.weight.grad = nn.Parameter(torch.cat((fc1_weight_grad,torch.mm(added_weights_post_layer.t(), growth_matrix).to(device)), dim=0))
        new_fc1.bias.grad = nn.Parameter(torch.cat((fc1_bias_grad, torch.zeros(num_neurons).to(device)), dim=0))

        # POST-LAYER

        num_in_2, num_out_2 = fc2.in_features, fc2.out_features
        new_fc2 = nn.Linear(in_features=num_in_2 + num_neurons, out_features=num_out_2).to(device)
        # Set parameters
        new_fc2.weight = nn.Parameter(torch.cat((fc2.weight, added_weights_post_layer), dim=1))
        new_fc2.bias = fc2.bias
        # Set gradients
        new_fc2.weight.grad = torch.cat((fc2_weight_grad, torch.zeros(num_out_2, num_neurons).to(device)), dim=1)
        new_fc2.bias.grad = fc2.bias.grad

    return new_fc1, new_fc2


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

        # Initialize the layers
        self.fc1 = nn.Linear(num_inputs,num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_outputs)
        
        self.activation = torch.sigmoid ## nn.ReLU()
        
    def forward(self, x) :
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        
        return x
    
    def add_neurons (self, layer_name, fc1_weight_grad, fc1_bias_grad, fc2_weight_grad, num_neurons, device, init_name, growth_matrix=None, c=None) :
        if layer_name == "fc1" :
            self.fc1, self.fc2 = add_neurons(self.fc1, self.fc2, 
                                             fc1_weight_grad, fc1_bias_grad, fc2_weight_grad,
                                             num_neurons, device, init_name, growth_matrix, c)
        
        elif layer_name == "fc2" :
            self.fc2, self.fc3 = add_neurons(self.fc2, self.fc3, 
                                             fc1_weight_grad, fc1_bias_grad, fc2_weight_grad,
                                             num_neurons, device, init_name, growth_matrix, c)