global_seed = 88
save_results = True
# Parameters specfific to the benchmark
benchmark_settings = {"benchmark_name" : "pMNIST_via_torch",
                      "difficulty" : "standard",
                      "num_tasks" : 2,
                      "train_percentage" : 0.2,
                      "num_val_benchmarks" : 1,
                      "batch_size" : 128}

# Parameters specific to the method
method_settings = {"method_name" : "GroHess",
                   "grow_from" : "output",
                   "hessian_percentile" : 92,
                   "grad_percentile" : 92,
                   "num_inputs" : 28*28,
                   "num_hidden_root" : 200,
                   "num_outputs" : 10,
                   "loss_name" : "CE",
                   "optimizer_name" : "Adam"}

# Parameters specific to HPO
HPO_settings = {"HPO_name" : "greedy_HPO",
                "n_trials" : 2,
                "lr" : (5e-5, 2e-3),
                "num_epochs" : (1,3),
                #"ewc_lambda" : (400,400)
                #"lwf_alpha" : (0.1, 0.9),
                #"lwf_temperature" : (1, 3),
                }



# IMPORTS
import sys
import os
import numpy as np
import warnings
import subprocess
import inspect
import torch
import pickle
import json
import ast
warnings.filterwarnings('ignore')

sys.path.append("Methods/" + method_settings["method_name"])
sys.path.append("HPO_lib")
sys.path.append("HPO_lib/benchmark_loaders")


path = os.path.dirname(os.path.abspath("__file__"))
data_path = path + "/data"



from HPO_lib.abstract_torch import get_device
from HPO_lib.get_benchmarks_for_script import get_benchmarks
from HPO_lib.validation import validate
from HPO_lib.save_and_load_results import save

from lib.method import initialize_model
from lib.method import train
from test_model import test
try :
    from lib.method import initialize_training
except :
    pass


# SET DEVICE
device = get_device(1)


# GET BENCHMARKS
benchmarks_list = get_benchmarks(benchmark_settings, global_seed)
benchmark = benchmarks_list[0]

"""
with open(f"logs/source_1.txt", "w") as f :
    train_targets = "train_1_targets : " + str(next(iter(benchmark[0][0]))[1][:10]) + "\n"
    train_inputs = "train_1_inputs : " + str(next(iter(benchmark[0][0]))[0][0][:10]) + "\n"
    val = "val_pendant_source: " + str(next(iter(benchmark[1][1]))[1][:10]) + "\n"
    f.write(train_targets)
    f.write(train_inputs)
    f.write(val)

with open(f"logs/source_2.txt", "w") as f :
    train_targets = "train_2_targets : " + str(next(iter(benchmark[0][1]))[1][:10]) + "\n"
    train_inputs = "train_2_inputs : " + str(next(iter(benchmark[0][1]))[0][0][:10]) + "\n"
    val = "val_pendant_source: " + str(next(iter(benchmark[1][1]))[1][:10]) + "\n"
    f.write(train_targets)
    f.write(train_inputs)
    f.write(val)



print("train_1_targets :", next(iter(benchmark[0][0]))[1][:10])
print("train_1_inputs :", next(iter(benchmark[0][0]))[0][0][:10])
print("train_2 :", next(iter(benchmark[0][1]))[1][:10])
print("train_2_inputs :", next(iter(benchmark[0][1]))[0][0][:10])

print("val :", next(iter(benchmark[1][1]))[1][:10])
"""

def retrain_and_save_with_best_HPs (model, params, method_settings, best_params, train_loader, device, global_seed) :
    # Get best HPs
    best_HPs = {}
    try :
        lr = best_params["lr"]
        best_HPs["lr"] = lr
    except :
        pass
    try :
        num_epochs = best_params["num_epochs"]
        best_HPs["num_epochs"] = num_epochs
    except :
        pass
    try :
        ewc_lambda = best_params["ewc_lambda"]
        best_HPs["ewc_lambda"] = ewc_lambda
    except :
        pass
    try :
        lwf_alpha = best_params["lwf_alpha"]
        best_HPs["lwf_alpha"] = lwf_alpha
    except :
        pass
    try :
        lwf_temperature = best_params["lwf_temperature"]
        best_HPs["lwf_temperature"] = lwf_temperature
    except :
        pass

    print(best_HPs)

    # Train
    if method_settings["method_name"] == "GroHess" :
        hessian_masks, overall_masks, _, _ = train(model, method_settings, params, best_HPs, train_loader, device, global_seed, verbose=2)
        return hessian_masks, overall_masks
    if method_settings["method_name"] == "EWC" :
        ewc = train(model, method_settings, params, best_HPs, train_loader, device, global_seed, verbose=2)
        return ewc
    if method_settings["method_name"] in ["EWC", "LwF", "Naive baseline"] :
        train(model, method_settings, params, best_HPs, train_loader, device, global_seed, verbose=2)


def call_greedy_HPO(HPO_settings, method_settings, benchmark_settings, benchmark, device, global_seed) :

    # Unpack train and test loaders
    try :
        train_loaders_list = benchmark[0].train_stream
    except :
        train_loaders_list = benchmark[0]
    test_loaders_list = benchmark[2]

    # Initialize model
    model = initialize_model(method_settings, global_seed).to(device)

    # Intialize HPO
    if method_settings["method_name"] == "GroHess" :
        hessian_masks, overall_masks = initialize_training(model, method_settings)
    best_params_list = []
    num_tasks = benchmark_settings["num_tasks"]
    test_accs_matrix = np.zeros((num_tasks, num_tasks))
    output = None

    for task_number in range(0, num_tasks) :

        train_loader = train_loaders_list[task_number]
        
        def call_script_task(hessian_masks, overall_masks, task_number, global_seed, method_settings, output, benchmark_settings, model, HPO_settings, device):
            signature = inspect.signature(call_script_task)
            names = [param.name for param in signature.parameters.values()]
            values = locals()
            dic = {name: values[name] for name in names}
            for key, value in dic.items() :
                try :
                    with open(f'logs/{key}.pkl', 'wb') as f:
                        pickle.dump(value, f)
                except :
                    torch.save(value, f'logs/{key}.pt')
                    
            best_params = subprocess.run(["python", "script_task.py"], 
                            input=json.dumps(names).encode(),
                            capture_output=True,
                            check=True)
            return best_params.stdout
        
        if method_settings["method_name"] == "GroHess" :
            best_params = call_script_task(hessian_masks, overall_masks, task_number, global_seed, method_settings, output, benchmark_settings, model, HPO_settings, device).decode()
        else :
            best_params = call_script_task(task_number, global_seed, method_settings, output, benchmark_settings, model, HPO_settings, device).decode()
        
        print("Voici les best params :", best_params)
        best_params = ast.literal_eval(best_params)

        # Retrain and save a model with the best params
        best_params_list += [best_params]
        if method_settings["method_name"] == "GroHess" :
            if output is not None :
                hessian_masks, overall_masks = output
            is_first_task = True if task_number==0 else False
            params = {"hessian_masks" : hessian_masks, "overall_masks" : overall_masks, "is_first_task" : is_first_task}
        if method_settings["method_name"] in ["EWC", "LwF"] :
            params = {"batch_size" : benchmark_settings["batch_size"]}
        output = retrain_and_save_with_best_HPs(model, params, method_settings, best_params, train_loader, device, global_seed) 
        

        # Test on each task
        for j in range(num_tasks) :
            test_accs_matrix[task_number,j] = round(test(model, test_loaders_list[j], device),2)
        
    return test_accs_matrix, best_params_list


test_accs_matrix, best_params_list = call_greedy_HPO(HPO_settings, method_settings, benchmark_settings, benchmark, device, global_seed)


print(test_accs_matrix)
print(best_params_list)


val_accs_matrix = validate(HPO_settings, benchmarks_list, benchmark_settings, method_settings, best_params_list, device, global_seed)


save(test_accs_matrix, best_params_list, val_accs_matrix, HPO_settings, method_settings, benchmark_settings, save_results)