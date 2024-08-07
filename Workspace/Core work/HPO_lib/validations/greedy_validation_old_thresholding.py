from lib.method import initialize_model
from lib.method import train
from test_model import test
try :
    from lib.method import initialize_training
except :
    pass
import numpy as np


def retrain_one_task (model, params, method_settings, best_params, train_loader, device, global_seed) :
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

    # Train
    if method_settings["method_name"] == "GroHess" :
        hessian_masks, overall_masks, _, _ = train(model, method_settings, params, best_HPs, train_loader, device, global_seed, verbose=2)
        return hessian_masks, overall_masks
    if method_settings["method_name"] == "EWC" :
        ewc = train (model, method_settings, params, best_HPs, train_loader, device, global_seed, verbose=0)
        return ewc
    if method_settings["method_name"] in ["LwF", "Naive baseline"] :
        train(model, method_settings, params, best_HPs, train_loader, device, global_seed, verbose=2)



def train_with_best_params (method_settings, benchmark_settings, best_params_list, benchmark, num_tasks, device, global_seed) :
    # Initialize model
    benchmark_model = initialize_model(method_settings, global_seed).to(device)

    # Intialize training
    output = None
    try :
        train_loaders_list = benchmark[0].train_stream
    except :
        train_loaders_list = benchmark[0]
    if method_settings["method_name"] == "GroHess" :
        hessian_masks, overall_masks = initialize_training(benchmark_model, method_settings)
    elif method_settings["method_name"] == "EWC" :
        output = initialize_training(benchmark_model, method_settings, benchmark_settings, device)

    for task_number in range(num_tasks) :

        # Verbose
        print("\n" + "-"*50)
        print(f"LEARNING TASK {task_number+1}")

        # Retrain and save a model with the best params
        params = {}
        train_loader = train_loaders_list[task_number]
        if method_settings["method_name"] == "GroHess" :
            if output is not None :
                hessian_masks, overall_masks = output
            is_first_task = True if task_number==0 else False
            params = {"hessian_masks" : hessian_masks, "overall_masks" : overall_masks, "is_first_task" : is_first_task}
        if method_settings["method_name"] in ["EWC"] :
            params = {"ewc" : output}
        if method_settings["method_name"] in ["LwF"] :
            params = {"batch_size" : benchmark_settings["batch_size"]}
        output = retrain_one_task(benchmark_model, params, method_settings, best_params_list[task_number], train_loader, device, global_seed) 
    
    return benchmark_model


def greedy_validate(benchmarks_list, benchmark_settings, method_settings, best_params_list, device, global_seed) :

    num_val_benchmarks = benchmark_settings["num_val_benchmarks"]
    num_tasks = benchmark_settings["num_tasks"]

    # Initialize the matrix to store the validation accuracies
    val_accs_matrix = np.zeros((num_val_benchmarks, num_tasks))

    # Train on each benchmark
    for i in range(1, num_val_benchmarks+1) :

        # Verbose
        print("\n" + "="*50)
        print(f"BENCHMARK {i}")

        # Train model with best params obtained through HPO on benchmark 0
        benchmark = benchmarks_list[i]
        benchmark_model = train_with_best_params(method_settings, benchmark_settings, best_params_list, benchmark, num_tasks, device, global_seed+1)

        # Test on each task
        test_loaders_list = benchmarks_list[i][2]
        for j in range(num_tasks) :
            val_accs_matrix[i-1,j] = round(test(benchmark_model, test_loaders_list[j], device),2)

    return val_accs_matrix