import csv
import numpy as np

def save(test_accs_matrix, best_params_list, val_accs_matrix, HPO_settings, method_settings, benchmark_settings, save_results) :
    
    if save_results :
        # Check if all required settings are present
        try :
            HPO_name = HPO_settings["HPO_name"]
        except ValueError :
            print("One or more of the required settings to visualize are missing. Please check the HPO_settings.")
        
        try :
            method_name = method_settings["method_name"]
        except ValueError :
            print("One or more of the required settings to visualize are missing. Please check the method_settings.")
        
        try :
            benchmark_name = benchmark_settings["benchmark_name"]
            difficulty = benchmark_settings["difficulty"]
        except ValueError :
            print("One or more of the required settings to visualize are missing. Please check the benchmark_settings.")

        test_accs_path = f'Results/test_accs_matrix_{HPO_name}_{method_name}_{benchmark_name}_{difficulty}.csv'
        with open(test_accs_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(test_accs_matrix)

        best_params_path = f'Results/best_params_{HPO_name}_{method_name}_{benchmark_name}_{difficulty}.csv'
        with open(best_params_path, mode='w', newline='', encoding='utf-8') as file:
            fieldnames = best_params_list[0].keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(best_params_list)
        
        val_accs_path = f'Results/val_accs_matrix_{HPO_name}_{method_name}_{benchmark_name}_{difficulty}.csv'
        with open(val_accs_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(val_accs_matrix)



def load(HPO_settings, method_settings, benchmark_settings) :
    
    # Check if all required settings are present
    try :
        HPO_name = HPO_settings["HPO_name"]
    except ValueError :
        print("One or more of the required settings to visualize are missing. Please check the HPO_settings.")
    
    try :
        method_name = method_settings["method_name"]
    except ValueError :
        print("One or more of the required settings to visualize are missing. Please check the method_settings.")
    
    try :
        benchmark_name = benchmark_settings["benchmark_name"]
        difficulty = benchmark_settings["difficulty"]
    except ValueError :
        print("One or more of the required settings to visualize are missing. Please check the benchmark_settings.")

    test_accs_path = f'Results/test_accs_matrix_{HPO_name}_{method_name}_{benchmark_name}_{difficulty}.csv'
    with open(test_accs_path, 'r', newline='', encoding='utf-8') as f:
        test_accs_matrix = []
        reader = csv.reader(f)
        for row in reader:
            test_accs_matrix.append([float(element) for element in row])
        test_accs_matrix = np.array(test_accs_matrix)

    best_params_path = f'Results/best_params_{HPO_name}_{method_name}_{benchmark_name}_{difficulty}.csv'
    with open(best_params_path, mode='r', newline='', encoding='utf-8') as file:
        best_params_list = []
        reader = csv.DictReader(file)
        for row in reader:
            best_params_list.append(dict(row))

    val_accs_path = f'Results/val_accs_matrix_{HPO_name}_{method_name}_{benchmark_name}_{difficulty}.csv'
    with open(val_accs_path, 'r', newline='', encoding='utf-8') as f:
        val_accs_matrix = []
        reader = csv.reader(f)
        for row in reader:
            val_accs_matrix.append([float(element) for element in row])
        val_accs_matrix = np.array(val_accs_matrix)
    
    return test_accs_matrix, best_params_list, val_accs_matrix