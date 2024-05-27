import numpy as np

def test (model, loader, task, batch_size, device) :
    batch_accs_list = []
    for batch in iter(loader) :
        data, targets = iter(batch)
        data, targets = data.to(device), targets.to(device)
        output, _ = model(data.view(batch_size, -1))
        _, idx = output.sum(dim=0).max(1)
        if task == "task_2" :
            targets -= 5
        batch_acc = np.mean((targets == idx).detach().cpu().numpy())
        batch_accs_list.append(batch_acc)

    return round(np.mean(batch_accs_list)*100,2)


def test_single_class (model, loader, batch_size, device) :
    batch_accs_list = []
    for batch in iter(loader) :
        data, targets = iter(batch)
        data, targets = data.to(device), targets.to(device)
        output, _ = model(data.view(batch_size, -1))
        _, idx = output.sum(dim=0).max(1)
        batch_acc = np.mean((targets == idx).detach().cpu().numpy())
        batch_accs_list.append(batch_acc)

    return round(np.mean(batch_accs_list)*100,2)


def test_ICL (model, loaders, batch_size, device) :
    task_accs_list = []
    for loader in loaders :
        task_acc = test_single_class (model, loader, batch_size, device)
        task_accs_list.append(task_acc)
    return round(np.mean(task_accs_list),2)