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


# SET DEVICE
device = get_device(1)


# GET BENCHMARKS
benchmarks_list = get_benchmarks(benchmark_settings, global_seed)
benchmark = benchmarks_list[0]



#test_accs_matrix, best_params_list = run_HPO(HPO_settings, method_settings, benchmark_settings, benchmarks_list[0], device, global_seed)

def is_pytorch_object(obj):
    pytorch_classes = (
        torch.Tensor,
        torch.nn.Module,
        torch.optim.Optimizer,
        torch.utils.data.dataloader.DataLoader
    )
    return isinstance(obj, pytorch_classes) 

import matplotlib.pyplot as plt
from torchvision import transforms

def save_transform(key, value) :
    for i, loader in enumerate(value) :
        try :
            #im = loader.dataset.dataset.dataset.data[0]
            #t_im = loader.dataset.dataset.dataset.transform(im.numpy())
            #print(loader.dataset.dataset.dataset.transform)
            # save the transformed image
            #print(type(t_im.numpy()))
            #pim = plt.imshow(t_im.numpy().reshape(28,28))
            #plt.savefig(f'logs/{key}_image_{i}.png')

            print(loader.dataset.dataset.dataset.transform)
            #print(dir(loader.dataset.dataset.dataset))

            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            print(type(transform))

            #torch.save(loader.dataset.dataset.dataset.data, f'logs/{key}_dataset_{i}.pt')
            #torch.save(loader.dataset.dataset.dataset.targets, f'logs/{key}_targets_{i}.pt')
            #torch.save(loader.dataset.dataset.indices, f'logs/{key}_indices_{i}.pt')
            #torch.save(loader.dataset.dataset.dataset.transform, f'logs/{key}_transform_{i}.pt')
            torch.save(value, f'logs/{key}_{i}.pt')
        except ValueError :
            print("Nan ça veut vraiment pas")

def call_greedy_HPO(HPO_settings, method_settings, benchmark_settings, benchmark, device, global_seed) :

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
        
        def call_script_task(hessian_masks, overall_masks, task_number, global_seed, train_loaders_list, val_loaders_list, method_settings, output, benchmark_settings, model, HPO_settings, device):
            signature = inspect.signature(call_script_task)
            names = [param.name for param in signature.parameters.values()]
            values = locals()
            dic = {name: values[name] for name in names}
            for key, value in dic.items() :
                if is_pytorch_object(value) :
                    torch.save(value, f'logs/{key}.pt')
                else :
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
            best_params = call_script_task(hessian_masks, overall_masks, task_number, global_seed, train_loaders_list, val_loaders_list, method_settings, output, benchmark_settings, model, HPO_settings, device)
        else :
            best_params = call_script_task(task_number, global_seed, train_loaders_list, val_loaders_list, method_settings, output, benchmark_settings, model, HPO_settings, device)
        
        print("Voici les best params :", best_params.decode())

        # Retrain and save a model with the best params
        #best_params = study.best_trial.params
        #best_params_list += [best_params]
        #if method_settings["method_name"] == "GroHess" :
        #    if output is not None :
        #        hessian_masks, overall_masks = output
        #    is_first_task = True if task_number==0 else False
        #    params = {"hessian_masks" : hessian_masks, "overall_masks" : overall_masks, "is_first_task" : is_first_task}
        #if method_settings["method_name"] in ["EWC", "LwF"] :
        #    params = {"batch_size" : benchmark_settings["batch_size"]}
        #output = retrain_and_save_with_best_HPs(model, params, method_settings, best_params, train_loader, device, global_seed) 
        

        # Test on each task
        #for j in range(num_tasks) :
        #    test_accs_matrix[task_number,j] = round(test(model, test_loaders_list[j], device),2)

        #optuna.delete_study(storage=storage, study_name=f"Search number {task_number+1}")
        
    return #test_accs_matrix, best_params_list


call_greedy_HPO(HPO_settings, method_settings, benchmark_settings, benchmark, device, global_seed)


    #val_accs_matrix = validate(HPO_settings, benchmarks_list, benchmark_settings, method_settings, best_params_list, device, global_seed)



    #save(test_accs_matrix, best_params_list, val_accs_matrix, HPO_settings, method_settings, benchmark_settings, save_results)