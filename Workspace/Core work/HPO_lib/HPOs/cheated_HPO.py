from lib.method import initialize_model
from lib.method import train
from test_model import test
try :
    from lib.method import initialize_training
except :
    pass
import numpy as np
import optuna
from functools import partial
import copy


def objective(num_tasks, HPO_settings, method_settings, benchmark_settings, train_loaders_list, val_loaders_list, device, global_seed, trial) :

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
     
    # Initialize model
    model = initialize_model(method_settings, global_seed).to(device)

    # Initialize training
    if method_settings["method_name"] == "GroHess" :
        overall_masks = initialize_training(model, method_settings)
    output = None

    for task_number in range(0, num_tasks) :
        params = {}
        train_loader = train_loaders_list[task_number]
        if method_settings["method_name"] == "GroHess" :
            if output is not None :
                overall_masks = output[0]
            is_first_task = True if task_number==0 else False
            params = {"overall_masks" : overall_masks, "is_first_task" : is_first_task}

        if method_settings["method_name"] in ["EWC", "LwF"] :
            params = {"batch_size" : benchmark_settings["batch_size"]}

        # Train
        output = train(model, method_settings, params, HPs, train_loader, device, global_seed)

    # Test
    test_accs = np.zeros(num_tasks)
    for j in range(num_tasks) :
        test_acc = test(model, val_loaders_list[j], device)
        test_accs[j] = test_acc
    
    # Compute score
    score = np.mean(test_accs)
    
    return score



def retrain_and_save_with_best_HPs (num_tasks, method_settings, benchmark_settings, best_params, train_loaders_list, test_loaders_list, device, global_seed) :
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

    # Initialize output
    test_accs_matrix = np.zeros((num_tasks, num_tasks))

    # Initialize model
    model = initialize_model(method_settings, global_seed).to(device)

    # Initialize training
    if method_settings["method_name"] == "GroHess" :
        overall_masks = initialize_training(model, method_settings)
    output = None

    for task_number in range(0, num_tasks) :
        params = {}
        train_loader = train_loaders_list[task_number]
        if method_settings["method_name"] == "GroHess" :
            if output is not None :
                overall_masks = output[0]
            is_first_task = True if task_number==0 else False
            params = {"overall_masks" : overall_masks, "is_first_task" : is_first_task}

        if method_settings["method_name"] in ["EWC", "LwF"] :
            params = {"batch_size" : benchmark_settings["batch_size"]}

        # Train
        output = train(model, method_settings, params, best_HPs, train_loader, device, global_seed, verbose=2)

        # Test on each task
        for j in range(num_tasks) :
            test_accs_matrix[task_number,j] = round(test(model, test_loaders_list[j], device),2)
    
    return test_accs_matrix



def call_cheated_HPO(HPO_settings, method_settings, benchmark_settings, benchmark, device, global_seed) :
    
    # Unpack loaders
    try :
        train_loaders_list = benchmark[0].train_stream
    except :
        train_loaders_list = benchmark[0]
    val_loaders_list, test_loaders_list = benchmark[1:]

    # Intialize HPO
    best_params_list = []
    num_tasks = benchmark_settings["num_tasks"]

    # Perform HPO
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(storage=storage,
                                study_name=f"Search",
                                sampler=optuna.samplers.TPESampler(seed=global_seed),
                                direction = "maximize")
    
    partial_objective = partial(objective, num_tasks, HPO_settings, method_settings, benchmark_settings, train_loaders_list, val_loaders_list, device, global_seed)
    study.optimize(partial_objective,
                n_trials=HPO_settings["n_trials"],
                timeout=3600)

    # Retrain and save a model with the best params
    best_params = study.best_trial.params
    best_params_list += [best_params]
    test_accs_matrix = retrain_and_save_with_best_HPs(num_tasks, method_settings, benchmark_settings, best_params, train_loaders_list, test_loaders_list, device, global_seed) 
    
    return test_accs_matrix, best_params_list