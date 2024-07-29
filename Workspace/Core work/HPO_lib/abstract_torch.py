import torch


def get_device(i=0) :
    if i == "cpu" :
        device = torch.device("cpu")
    else :
        if torch.cuda.device_count() > 1 :
            device = torch.device(f"cuda:{i}")
        else :
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    return device

def get_loss(loss_name) :
    if loss_name == "CE":
        loss = torch.nn.CrossEntropyLoss()
    elif loss_name == "MSE":
        loss = torch.nn.MSELoss()
    return loss

def get_optimizer(optimizer_name, model, lr=5e-3) :
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)#, betas=(1e-8, 0.999))
    return optimizer