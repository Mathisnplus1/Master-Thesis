import torch


if torch.cuda.is_available():
    device = torch.device('cuda')
    my_device = 'cuda:0'

    my_device_0 = 'cuda:0'
    if torch.cuda.device_count() > 1:
        my_device_1 = 'cuda:1'
    else:
        my_device_1 = 'cuda:0'
else:
    device = torch.device('cpu')
    my_device = 'cpu'
    my_device_0 = 'cpu'
    my_device_1 = 'cpu'
