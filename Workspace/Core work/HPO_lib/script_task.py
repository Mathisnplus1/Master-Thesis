import sys
import optuna
from functools import partial
import torch
import json
import pickle


def is_pytorch_object(obj):
    pytorch_classes = (
        torch.Tensor,
        torch.nn.Module,
        torch.optim.Optimizer
    )
    return isinstance(obj, pytorch_classes)


inputs = json.loads(sys.stdin.read())
names, train_loaders_list, val_loaders_list = inputs[0], inputs[1], inputs[2]
print(json.dumps("THROUGH"))

for name in names :
    try :
        try :
            with open(f'logs/{name}.pkl', 'rb') as f:
                locals()[name] = pickle.load(f)
        except :
            locals()[name] = torch.load(f'logs/{name}.pt')
    except :
        pass

print(json.dumps(global_seed))


def objective(model, task_number, HPO_settings, params, method_settings, train_loader, val_loaders_list, device, global_seed, trial) :
    trial_result = greedy_objective(model, task_number, HPO_settings, params, method_settings, train_loader, val_loaders_list, device, global_seed, trial)

    score = trial_result.stdout

    return score


# Verbose
print("\n" + "-"*50)
print(f"LEARNING TASK {task_number+1}")

# Perform HPO
storage = optuna.storages.InMemoryStorage()
study = optuna.create_study(storage=storage,
                            study_name = f"Search number {task_number+1}",
                            sampler = optuna.samplers.TPESampler(seed=global_seed),
                            #sampler = optuna.samplers.RandomSampler(seed=global_seed),
                            direction = "maximize")
params = {}
#train_loader = train_loaders_list[task_number]
#if method_settings["method_name"] == "GroHess" :
#    if output is not None :
#        hessian_masks, overall_masks = output
#    is_first_task = True if task_number==0 else False
#    params = {"hessian_masks" : hessian_masks, "overall_masks" : overall_masks, "is_first_task" : is_first_task}
#if method_settings["method_name"] in ["EWC", "LwF"] :
#    params = {"batch_size" : benchmark_settings["batch_size"]}
#partial_objective = partial(objective, model, task_number, HPO_settings, params, method_settings, train_loader, val_loaders_list, device, global_seed)
#study.optimize(partial_objective,
#                n_jobs=1,
#                n_trials=HPO_settings["n_trials"],
#                timeout=3600)