import numpy as np
import torch
import torch.nn as nn


##################################
#### Function to add neurons #####
##################################


def add_neurons_from_input(new_mask, fc1, fc2, device) :

    line_mask = (new_mask.sum(axis=1) != 0)
    negative_mask = -1*(new_mask-1)
    num_neurons = line_mask.sum()

    # PRE-LAYER

    num_in_1, num_out_1 = fc1.in_features, fc1.out_features
    new_fc1 = nn.Linear(in_features=num_in_1, out_features=num_out_1+num_neurons).to(device)
    # Set parameters
    new_fc1.weight = nn.Parameter(torch.cat((fc1.weight, (fc1.weight*negative_mask)[line_mask].to(device))))
    new_fc1.bias = nn.Parameter(torch.cat((fc1.bias, (fc1.bias*line_mask)[line_mask].to(device))))
    # Set gradients
    new_fc1.weight.grad = nn.Parameter(torch.cat((fc1.weight.grad, (fc1.weight.grad*negative_mask)[line_mask].to(device)), dim=0))
    new_fc1.bias.grad = nn.Parameter(torch.cat((fc1.bias.grad, (fc1.bias.grad*line_mask)[line_mask].to(device)), dim=0))

    # POST-LAYER

    num_in_2, num_out_2 = fc2.in_features, fc2.out_features
    new_fc2 = nn.Linear(in_features=num_in_2+num_neurons, out_features=num_out_2).to(device)
    # Set parameters
    new_fc2.weight = nn.Parameter(torch.cat((fc2.weight, fc2.weight[:,line_mask].to(device)), dim=1))
    new_fc2.bias = nn.Parameter(fc2.bias.data)
    # Set gradients
    new_fc2.weight.grad = nn.Parameter(torch.cat((fc2.weight.grad, fc2.weight.grad[:,line_mask].to(device)), dim=1))
    new_fc2.bias.grad = nn.Parameter(fc2.bias.grad)

    return new_fc1, new_fc2

def add_neurons_from_output(new_mask, fc1, fc2, device) :

    column_mask = (new_mask.sum(axis=0) != 0)
    negative_mask = -1*(new_mask-1)
    num_neurons = column_mask.sum()

    # PRE-LAYER

    num_in_1, num_out_1 = fc1.in_features, fc1.out_features
    new_fc1 = nn.Linear(in_features=num_in_1, out_features=num_out_1+num_neurons).to(device)
    # Set parameters
    new_fc1.weight = nn.Parameter(torch.cat((fc1.weight, (fc1.weight)[column_mask].to(device))))
    #print("new_fc1.weight :", new_fc1.weight.shape)
    new_fc1.bias = nn.Parameter(torch.cat((fc1.bias, (fc1.bias*column_mask)[column_mask].to(device))))
    # Set gradients
    new_fc1.weight.grad = nn.Parameter(torch.cat((fc1.weight.grad, (fc1.weight.grad)[column_mask].to(device)), dim=0))
    new_fc1.bias.grad = nn.Parameter(torch.cat((fc1.bias.grad, (fc1.bias.grad*column_mask)[column_mask].to(device)), dim=0))

    # POST-LAYER

    num_in_2, num_out_2 = fc2.in_features, fc2.out_features
    new_fc2 = nn.Linear(in_features=num_in_2+num_neurons, out_features=num_out_2).to(device)
    # Set parameters
    new_fc2.weight = nn.Parameter(torch.cat((fc2.weight, ((fc2.weight*negative_mask)[:,column_mask]).to(device)), dim=1))
    new_fc2.bias = nn.Parameter(fc2.bias.data)
    # Set gradients
    new_fc2.weight.grad = nn.Parameter(torch.cat((fc2.weight.grad, ((fc2.weight.grad*column_mask)[:,column_mask]).to(device)), dim=1))
    new_fc2.bias.grad = nn.Parameter(fc2.bias.grad)

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
    
    def add_neurons(self, new_mask, layer_to_grow, grow_from, device) :
        if grow_from == "input" :
            if layer_to_grow == "fc1" :
                new_fc1, new_fc2 = add_neurons_from_input(new_mask, self.fc1, self.fc2, device)
            else :
                new_fc2, new_fc3 = add_neurons_from_input(new_mask, self.fc2, self.fc3, device)
        else :
            if layer_to_grow == "fc1" :
                new_fc1, new_fc2 = add_neurons_from_output(new_mask, self.fc1, self.fc2, device)
            else :
                new_fc2, new_fc3 = add_neurons_from_output(new_mask, self.fc2, self.fc3, device)
        try :
            self.fc1 = new_fc1
            self.fc2 = new_fc2
        except :
            self.fc2 = new_fc2
            self.fc3 = new_fc3