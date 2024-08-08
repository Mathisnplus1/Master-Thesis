import numpy as np
import copy
from lib.method import train
from test_model import test
import ctypes
import gc
import torch
#import pickle



def greedy_objective(model, task_number, HPO_settings, params, method_settings, train_loader, val_loaders_list, device, global_seed, trial) :

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
    try :
        tau = trial.suggest_float("tau", HPO_settings["tau"][0], HPO_settings["tau"][1])
        HPs["tau"] = tau
    except : 
        pass
     

    # Copy the model to perform HPO
    model_copy = copy.deepcopy(model)
    params_copy = copy.deepcopy(params)

    #if method_settings["method_name"] == "EWC" :
    #    ewc_copy = copy.deepcopy(params["ewc"])
    #    params["ewc"] = ewc_copy

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

    with open("logs/score.txt", "w") as f :
        f.write(str(score))
    
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                del obj
        except:
            pass

    all_vars = [var for var in locals().keys() if var[0] != '_']
    for var in all_vars :
        del (locals()[var])
    
    # Run garbage collection
    gc.collect()

    # Empty CUDA cache
    torch.cuda.empty_cache()