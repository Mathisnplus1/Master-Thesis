global_seed = 88
save_results = True
# Parameters specfific to the benchmark
benchmark_settings = {"benchmark_name" : "pMNIST_via_torch",
                      "difficulty" : "standard",
                      "num_tasks" : 3,
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
                "n_trials" : 1,
                "lr" : (5e-5, 2e-3),
                "num_epochs" : (3,3),
                #"ewc_lambda" : (400,400)
                #"lwf_alpha" : (0.1, 0.9),
                #"lwf_temperature" : (1, 3),
                }




import sys
import os
import numpy as np
import warnings
import subprocess
warnings.filterwarnings('ignore')

sys.path.append("Methods/" + method_settings["method_name"])
sys.path.append("HPO_lib")
sys.path.append("HPO_lib/benchmark_loaders")


path = os.path.dirname(os.path.abspath("__file__"))
data_path = path + "/data"



from HPO_lib.abstract_torch import get_device
from HPO_lib.get_benchmarks import get_benchmarks
from HPO_lib.validation import validate
from HPO_lib.save_and_load_results import save

from lib.method import initialize_model
from lib.method import train
from test_model import test
try :
    from lib.method import initialize_training
except :
    pass



device = get_device(0)



benchmarks_list = get_benchmarks(benchmark_settings, global_seed)
benchmark = benchmarks_list[0]



#test_accs_matrix, best_params_list = run_HPO(HPO_settings, method_settings, benchmark_settings, benchmarks_list[0], device, global_seed)

#def call_greedy_HPO(HPO_settings, method_settings, benchmark_settings, benchmark, device, global_seed) :

# Unpack loaders
try :
    train_loaders_list = benchmark[0].train_stream
except :
    train_loaders_list = benchmark[0]
val_loaders_list, test_loaders_list = benchmark[1:]

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

    subprocess.run(["python", "script_task.py"], 
                   input = inputs,
                   check = True,
                   capture_output = True)

    # Retrain and save a model with the best params
    best_params = study.best_trial.params
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

    #optuna.delete_study(storage=storage, study_name=f"Search number {task_number+1}")
    
return test_accs_matrix, best_params_list





val_accs_matrix = validate(HPO_settings, benchmarks_list, benchmark_settings, method_settings, best_params_list, device, global_seed)



save(test_accs_matrix, best_params_list, val_accs_matrix, HPO_settings, method_settings, benchmark_settings, save_results)