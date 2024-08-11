import sys
import os
import numpy as np
import copy
import ctypes
import gc
import torch
import pickle
import json


with open("logs/method_settings.pkl", "rb") as f :
    method_settings = pickle.load(f)


sys.path.append("Methods/" + method_settings["method_name"])
sys.path.append("HPO_lib")
sys.path.append("HPO_lib/benchmark_loaders")


path = os.path.dirname(os.path.abspath("__file__"))
data_path = path + "/data"



from HPO_lib.abstract_torch import get_device
from HPO_lib.get_benchmarks_for_script import get_loaders_for_trial
from HPO_lib.validation import validate
from HPO_lib.save_and_load_results import save

from lib.method_old_thresholding import initialize_model
from lib.method_old_thresholding import train
from test_model import test
try :
    from lib.method_old_thresholding import initialize_training
except :
    pass



def greedy_objective(model, task_number, HPO_settings, params, method_settings, train_loader, val_loaders_list, device, global_seed, HPs) :
    
    # Copy the model to perform HPO
    model_copy = copy.deepcopy(model)
    params_copy = copy.deepcopy(params)

    # Train
    _ = train(model_copy, method_settings, params_copy, HPs, train_loader, device, global_seed)

    if method_settings["method_name"] == "EWC" :
        model_copy = params_copy["ewc"].model

    # Test
    test_accs = np.zeros(task_number+1)
    for j in range(task_number+1) :
        test_acc = test(model_copy, val_loaders_list[j], device)
        test_accs[j] = test_acc
    
    # Compute score
    score = np.mean(test_accs)

    return score


names_to_retrieve = ["benchmark_settings",
                    "model",
                    "task_number",
                    "HPO_settings",
                    "params",
                    "method_settings",
                    "device",
                    "global_seed",
                    "HPs"]

names_to_retrieve = json.loads(sys.stdin.read())

for name in names_to_retrieve :
    try :
        try :
            with open(f'logs/{name}.pkl', 'rb') as f:
                locals()[name] = pickle.load(f)
        except :
            locals()[name] = torch.load(f'logs/{name}.pt')
    except :
        print(name)

train_loader, val_loaders_list = get_loaders_for_trial(benchmark_settings, task_number, global_seed)

"""
with open(f"logs/trial_{task_number}.txt", "w") as f :
    train_targets = f"train_{task_number}_targets : " + str(next(iter(train_loader))[1][:10]) + "\n"
    train_inputs = f"train_{task_number}_inputs : " + str(next(iter(train_loader))[0][0][:10]) + "\n"
    val = f"val_pendant_{task_number}: " + str(next(iter(val_loaders_list[1]))[1][:10]) + "\n"
    f.write(train_targets)
    f.write(train_inputs)
    f.write(val)
"""


score = greedy_objective(model, task_number, HPO_settings, params, method_settings, train_loader, val_loaders_list, device, global_seed, HPs)

print(score)