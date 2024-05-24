import numpy as np

import torch
import torch.nn as nn

import snntorch as snn


###########################
#### Define plain SNN #####
###########################

class SNN (nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta, num_steps=25):
        super().__init__()
        
        self.fc1 = nn.Linear(num_inputs,num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)
        self.num_steps = num_steps

    def forward(self, x) :
        # Initiamize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record the final layer
        spk2_rec = []
        mem2_rec = []
        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
    

##########################
#### Define LoRA SNN #####
##########################

def retrieve_model_weights(model, batch_size) :
    cur_size_out = 0
    for child in model.named_children() :
        layer_name = child[0]
        if layer_name[:2] == "fc" :
            cur_size_out = child[1].out_features
        elif layer_name[:3] == "lif" :
            layer = getattr(model, layer_name)
            mem1 = layer.init_leaky()
            cur1 = torch.zeros((batch_size, cur_size_out))
            spk1, mem1 = layer(cur1, mem1)
    
    return model

class LoRA_SNN(nn.Module):
    def __init__(self, lora_rank, alpha, path, num_inputs, num_hidden, num_outputs, beta, batch_size=128, num_steps=25):
        super(LoRA_SNN, self).__init__()
        
        # Load the pretrained model
        model = SNN(num_inputs, num_hidden, num_outputs, beta)
        self.model = retrieve_model_weights(model, batch_size)
        self.model.load_state_dict(torch.load(path))
        
        # Define LoRA hyperparameters
        self.lora_rank = lora_rank
        self.alpha = alpha
        
        # Define LoRA weights matrices for each layer in pretrained model
        self.l1_lora_A = nn.Parameter(torch.Tensor(self.model.fc1.in_features, lora_rank))
        self.l1_lora_B = nn.Parameter(torch.Tensor(lora_rank, self.model.fc1.out_features))
        
        self.l2_lora_A = nn.Parameter(torch.Tensor(self.model.fc2.in_features, lora_rank))
        self.l2_lora_B = nn.Parameter(torch.Tensor(lora_rank, self.model.fc2.out_features))
        
        # Initialization for LoRA layers
        nn.init.kaiming_uniform_(self.l1_lora_A, a=np.sqrt(5))
        # nn.init.xavier_normal_(self.l1_lora_A)#, mean=0.0, std=1.0)
        nn.init.zeros_(self.l1_lora_B)
        
        nn.init.kaiming_uniform_(self.l2_lora_A, a=np.sqrt(5))
        # nn.init.xavier_normal_(self.l2_lora_A)#, mean=0.0, std=1.0)
        nn.init.zeros_(self.l2_lora_B)
        
        # Freeze non-LoRA weights
        self.model.fc1.weight.requires_grad = False
        self.model.fc1.bias.requires_grad = False
        self.model.fc2.weight.requires_grad = False
        self.model.fc2.bias.requires_grad = False

        self.num_steps = num_steps
        
    def lora_linear(self, x, layer, lora_A, lora_B):
        h = torch.add(layer(x),self.alpha*torch.mm(torch.mm(x, lora_A), lora_B))
        return h
        
    def forward(self, x):
        # x = input.view(-1, 28*28)
        # Initiamize hidden states at t=0
        mem1 = self.model.lif1.init_leaky()
        mem2 = self.model.lif2.init_leaky()
        
        # Record the final layer
        spk2_rec = []
        mem2_rec = []
        for step in range(self.num_steps):
            # Apply LoRA layer 1
            cur1 = self.lora_linear(x, self.model.fc1, self.l1_lora_A, self.l1_lora_B)
            
            # Apply LIF layer 1 
            spk1, mem1 = self.model.lif1(cur1, mem1)
            
            # Apply LoRA layer 2
            cur2 = self.lora_linear(spk1, self.model.fc2, self.l2_lora_A, self.l2_lora_B)
            
            # Apply LIF layer 2
            spk2, mem2 = self.model.lif2(cur2, mem2)
            
            # Record the time step
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)