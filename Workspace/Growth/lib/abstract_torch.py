import torch


def get_device() :
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device

def get_loss(loss_name) :
    if loss_name == "CE":
        loss = torch.nn.CrossEntropyLoss()
    elif loss_name == "MSE":
        loss = torch.nn.MSELoss()
    return loss

def get_optimizer(optimizer_name, model, lr=5e-3) :
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer