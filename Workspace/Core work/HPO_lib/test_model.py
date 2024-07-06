import numpy as np

def get_batch_accuracy(model, data, targets):
    output = model(data.view(data.shape[0], -1))
    idx = output.argmax(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    return round(acc*100,2)

def test (model, loader, device) :
    batches = iter(loader)
    num_batches = len(batches)
    acc_sum = 0
    for (data, targets) in batches :
        acc_sum += get_batch_accuracy(model, data.to(device), targets.to(device))
    return acc_sum/num_batches