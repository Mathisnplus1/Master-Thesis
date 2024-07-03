import torch

def save(path, model) :
    torch.save(model.state_dict(), path)