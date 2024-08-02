import sys
import os
import optuna
from functools import partial
import torch
import json
import pickle
import subprocess


sys.path.append("Methods/" + "GroHess")
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


names = ["benchmark_settings",
         "device",
         "global_seed",
         "hessian_masks",
         "HPO_settings",
         "method_settings",
         "model",
         "output",
         "overall_masks",
         "task_number"]
names = json.loads(sys.stdin.read())


#model = torch.load(f'logs/model.pt')
#with open(f'logs/train_loaders_list.pkl', 'rb') as f:
#    train_loaders_list = pickle.load(f)
#train_loaders_list = torch.load(f'logs/train_loaders_list.pt')

for name in names :
    try :
        try :
            with open(f'logs/{name}.pkl', 'rb') as f:
                locals()[name] = pickle.load(f)
        except :
            locals()[name] = torch.load(f'logs/{name}.pt')
    except :
        print(name)


def objective(benchmark_settings, model, task_number, HPO_settings, params, method_settings, device, global_seed, trial) :
    # Set HPs
    HPs = {}
    try :
        lr = trial.suggest_float("lr", HPO_settings["lr"][0], HPO_settings["lr"][1])
        HPs["lr"] = lr
    except :
        pass
    try :
        num_epochs = trial.suggest_int("num_epochs", HPO_settings["num_epochs"][0], HPO_settings["num_epochs"][1])
        HPs["num_epochs"] = num_epochs
    except :
        pass
    try :
        ewc_lambda = trial.suggest_int("ewc_lambda", HPO_settings["ewc_lambda"][0], HPO_settings["ewc_lambda"][1])
        HPs["ewc_lambda"] = ewc_lambda
    except :
        pass
    try :
        lwf_alpha = trial.suggest_float("lwf_alpha", HPO_settings["lwf_alpha"][0], HPO_settings["lwf_alpha"][1])
        HPs["lwf_alpha"] = lwf_alpha
    except :
        pass
    try :
        lwf_temperature = trial.suggest_int("lwf_temperature", HPO_settings["lwf_temperature"][0], HPO_settings["lwf_temperature"][1])
        HPs["lwf_temperature"] = lwf_temperature
    except :
        pass
    
    names_to_retrieve = ["benchmark_settings", "model", "task_number", "HPO_settings", "params", "method_settings", "device", "global_seed", "HPs"]
    
    with open(f'logs/HPs.pkl', 'wb') as f:
        pickle.dump(HPs, f)
    
    trial_result = subprocess.run(["python", "script_trial.py"], 
                            input=json.dumps(names_to_retrieve).encode(),
                            capture_output=True,
                            check=True)

    score = float(trial_result.stdout.decode())

    #print(dir(trial))
    #print(score)

    return score

# Perform HPO
storage = optuna.storages.InMemoryStorage()
study = optuna.create_study(storage=storage,
                            study_name = f"Search number {task_number+1}",
                            sampler = optuna.samplers.TPESampler(seed=global_seed),
                            #sampler = optuna.samplers.RandomSampler(seed=global_seed),
                            direction = "maximize")
params = {}

if method_settings["method_name"] == "GroHess" :
    if output is not None :
        hessian_masks, overall_masks = output
    is_first_task = True if task_number==0 else False
    params = {"hessian_masks" : hessian_masks, "overall_masks" : overall_masks, "is_first_task" : is_first_task}
if method_settings["method_name"] in ["EWC", "LwF"] :
    params = {"batch_size" : benchmark_settings["batch_size"]}

# Save useful variables
with open(f'logs/params.pkl', 'wb') as f:
    pickle.dump(params, f)

partial_objective = partial(objective, benchmark_settings, model, task_number, HPO_settings, params, method_settings, device, global_seed)
study.optimize(partial_objective,
                n_jobs=1,
                n_trials=HPO_settings["n_trials"],
                timeout=3600)

best_params = study.best_trial.params

print(best_params)