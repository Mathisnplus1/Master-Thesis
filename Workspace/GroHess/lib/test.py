import numpy as np

def get_batch_accuracy(model, data, targets, batch_size):
    output = model(data.view(batch_size, -1))
    idx = output.argmax(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    return round(acc*100,2)

def test (model, loader, batch_size, device) :
    batches = iter(loader)
    num_batches = len(batches)
    acc_sum = 0
    for (data, targets) in batches :
        acc_sum += get_batch_accuracy(model, data.to(device), targets.to(device), batch_size)
    return acc_sum/num_batches