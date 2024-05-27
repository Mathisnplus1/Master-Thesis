import torch


def count_all_parameters(model) :
    # Count the parameters in the linear layers (which does not include parameters within LIF layers)
    lora_params = []
    for param_name, param in model.named_parameters():
        if not 'lora' in param_name:
            lora_params.append(param)
    num_lora_params = sum(torch.numel(param) for param in lora_params)

    return num_lora_params


def count_lora_parameters(model):
    # Count the parameters introduced by lora layers (trainable parameters only)
    lora_params = []
    for param_name, param in model.named_parameters():
        if 'lora' in param_name:
            lora_params.append(param)
    num_lora_params = sum(torch.numel(param) for param in lora_params)

    return num_lora_params