import os
from pMNIST_via_torch import get_task_loaders
from pMNIST_via_avalanche import PermutedMNIST


def get_benchmarks (benchmark_settings, global_seed, transform=None) :
    # Get benchmark settings
    try :
        benchmark_name = benchmark_settings["benchmark_name"]
        num_tasks = benchmark_settings["num_tasks"]
        num_val_benchmarks = benchmark_settings["num_val_benchmarks"]
        batch_size = benchmark_settings["batch_size"]
        train_percentage = benchmark_settings["train_percentage"]
        difficulty = benchmark_settings["difficulty"]
    except ValueError:
        print("One or more of the required settings to get the benchmarks are missing. Please check the benchmark_settings.")
    
    path = os.path.dirname(os.path.abspath("__file__"))
    data_path = path + "/data"

    benchmarks_list = []

    if benchmark_name == "pMNIST_via_avalanche" :
        random_seed_list = [i for i in range(num_val_benchmarks+1)]
        for i in range(num_val_benchmarks+1) :
            benchmark = PermutedMNIST(n_experiences=num_tasks, train_percentage=train_percentage, difficulty=difficulty, batch_size=batch_size, seed=random_seed_list[i], global_seed=global_seed)
            benchmarks_list += [benchmark]
    
    if benchmark_name == "pMNIST_via_torch" :
        permutation_random_seeds_list = [list(range(num_tasks*(i), num_tasks*(i+1))) for i in range(num_val_benchmarks+1)]

        for i in range(num_val_benchmarks+1) :
            permutation_random_seeds = permutation_random_seeds_list[i]
            train_loaders_list = []
            val_loaders_list = []
            test_loaders_list = []
            for random_seed in permutation_random_seeds :
                try :
                    loaders = get_task_loaders(data_path, batch_size, global_seed, random_seed, train_percentage, difficulty, False, transform)
                except :
                    loaders = get_task_loaders(data_path, batch_size, global_seed, random_seed, train_percentage, difficulty, True, transform)
                train_loader, val_loader, test_loader = loaders
                train_loaders_list += [train_loader]
                val_loaders_list += [val_loader]
                test_loaders_list += [test_loader]
            benchmarks_list += [(train_loaders_list, val_loaders_list, test_loaders_list)]
    
    return benchmarks_list