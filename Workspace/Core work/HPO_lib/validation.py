from lib.method import initialize_model
from lib.method import train
from test_model import test
try :
    from lib.method import initialize_training
except :
    pass
import numpy as np


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
    overall_masks, _, _ = train(model, method_settings, params, best_HPs, train_loader, device, global_seed, verbose=2)
    
    return overall_masks



def train_with_best_params (method_settings, best_params_list, train_loaders_list, num_tasks, device, global_seed) :
    # Initialize model
    benchmark_model = initialize_model(method_settings, global_seed).to(device)

    # Intialize training
    if method_settings["method_name"] == "GroHess" :
        overall_masks = initialize_training(benchmark_model, method_settings)

    for task_number in range(num_tasks) :

        # Verbose
        print("\n" + "-"*50)
        print(f"LEARNING TASK {task_number+1}")

        # Retrain and save a model with the best params
        is_first_task = True if task_number==0 else False
        params = {"overall_masks" : overall_masks, "is_first_task" : is_first_task}
        overall_masks = retrain_and_save_with_best_HPs(benchmark_model, params, method_settings, best_params_list[task_number], train_loaders_list[task_number], device, global_seed) 
    
    return benchmark_model


def validate(benchmarks_list, method_settings, best_params_list, device, global_seed) :

    num_val_benchmarks = len(benchmarks_list)-2
    num_tasks = len(benchmarks_list[0][0])

    # Initialize the matrix to store the validation accuracies
    val_accs_matrix = np.zeros((num_val_benchmarks+1, num_tasks))

    # Train on each benchmark
    for i in range(1, num_val_benchmarks+2) :

        # Verbose
        print("\n" + "="*50)
        print(f"BENCHMARK {i-1}")

        # Train model with best params obtained through HPO on benchmark 0
        train_loaders_list = benchmarks_list[i][0]
        benchmark_model = train_with_best_params(method_settings, best_params_list, train_loaders_list, num_tasks, device, global_seed+1)

        # Test on each task
        test_loaders_list = benchmarks_list[i][2]
        for j in range(num_tasks) :
            val_accs_matrix[i-1,j] = round(test(benchmark_model, test_loaders_list[j], device),2)

    return val_accs_matrix