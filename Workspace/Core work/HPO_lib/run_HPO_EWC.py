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


def objective(model, task_number, HP_settings, params, method_settings, train_loader, val_loaders_list, device, global_seed, trial) :

    # Set HPs
    HPs = {}
    try :
        lr = trial.suggest_float("lr", HP_settings["lr"][0], HP_settings["lr"][1])
        HPs["lr"] = lr
    except :
        pass
    try :
        num_epochs = trial.suggest_int("num_epochs", HP_settings["num_epochs"][0], HP_settings["num_epochs"][1])
        HPs["num_epochs"] = num_epochs
    except :
        pass
    try :
        ewc_lambda = trial.suggest_int("ewc_lambda", HP_settings["ewc_lambda"][0], HP_settings["ewc_lambda"][1])
        HPs["ewc_lambda"] = ewc_lambda
    except :
        pass
     

    # Copy the model to perform HPO
    model_copy = copy.deepcopy(model)
    
    # Train
    _ = train(model_copy, method_settings, params, HPs, train_loader, device, global_seed)

    # Test
    test_accs = np.zeros(task_number+1)
    for j in range(task_number+1) :
        test_acc = test(model_copy, val_loaders_list[j], device)
        test_accs[j] = test_acc
    
    # Compute score
    score = np.mean(test_accs)
    
    return score



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

    # Train
    if method_settings["method_name"] == "GroHess" :
        overall_masks, _, _ = train(model, method_settings, params, best_HPs, train_loader, device, global_seed, verbose=2)
        return overall_masks
    
    if method_settings["method_name"] == "EWC" :
        train(model, method_settings, params, best_HPs, train_loader, device, global_seed, verbose=2)



def call_greedy_HPO(HPO_settings, method_settings, benchmark_settings, benchmark, device, global_seed) :
    
    # Unpack loaders
    train_loaders_list, val_loaders_list, test_loaders_list = benchmark

    # Initialize model
    model = initialize_model(method_settings, global_seed).to(device)

    # Intialize HPO
    if method_settings["method_name"] == "GroHess" :
        overall_masks = initialize_training(model, method_settings)
    best_params_list = []
    num_tasks = len(benchmark[0])
    test_accs_matrix = np.zeros((num_tasks, num_tasks))
    output = None

    for task_number in range(0, num_tasks) :

        # Verbose
        print("\n" + "-"*50)
        print(f"LEARNING TASK {task_number+1}")

        # Perform HPO
        storage = optuna.storages.InMemoryStorage()
        study = optuna.create_study(storage=storage,
                                    study_name=f"Search number {task_number+1}",
                                    sampler=optuna.samplers.TPESampler(seed=global_seed),
                                    direction = "maximize")
        if method_settings["method_name"] == "GroHess" :
            if output is not None :
                overall_masks = output
            is_first_task = True if task_number==0 else False
            params = {"overall_masks" : overall_masks, "is_first_task" : is_first_task}
            train_loader = train_loaders_list[task_number]

        if method_settings["method_name"] == "EWC" :
            params = {"batch_size" : benchmark_settings["batch_size"]}
            train_loader = benchmark.train_stream[0]
        partial_objective = partial(objective, model, task_number, HPO_settings, params, method_settings, train_loader, val_loaders_list, device, global_seed)
        study.optimize(partial_objective,
                    n_trials=HPO_settings["n_trials"],
                    timeout=3600)

        # Retrain and save a model with the best params
        best_params = study.best_trial.params
        best_params_list += [best_params]
        output = retrain_and_save_with_best_HPs(model, params, method_settings, best_params, train_loader, device, global_seed) 
        
        # Test on each task
        for j in range(num_tasks) :
            test_accs_matrix[task_number,j] = round(test(model, test_loaders_list[j], device),2)
    
    return test_accs_matrix, best_params_list, model



def run_HPO(HPO_settings, method_settings, benchmark_settings, benchmark, device, global_seed) :
    if HPO_settings["HPO_name"] == "greedy_HPO" :
        return call_greedy_HPO(HPO_settings, method_settings, benchmark_settings, benchmark, device, global_seed)