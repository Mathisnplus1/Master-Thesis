import os
import csv
import numpy as np
import re
import ast



def extract_benchmark_settings(input_string):
    # Regular expression to find the benchmark_settings dictionary
    pattern_benchmark_settings = r'benchmark_settings\s*:\s*(\{.*?\})\n'
    pattern_method_settings = r'method_settings\s*:\s*(\{.*?\})\n'
    pattern_HPO_settings = r'HPO_settings\s*:\s*\{[^}]*"HPO_name"\s*:\s*"([^"]+)"'
    #pattern_save_results = r'save_results\s*:\s*(True|False)'
    
    # Search the input string for the pattern
    match_1 = re.search(pattern_benchmark_settings, input_string, re.DOTALL)
    match_2 = re.search(pattern_method_settings, input_string, re.DOTALL)
    match_3 = re.search(pattern_HPO_settings, input_string, re.DOTALL)
    #match_4 = re.search(pattern_save_results, input_string)

    if match_1 and match_2 and match_3 :#and match_4:
        benchmark_settings = ast.literal_eval(match_1.group(1))
        method_settings = ast.literal_eval(match_2.group(1))
        HPO_name = match_3.group(1)
        #save_results = bool(match_4.group(1))
        return benchmark_settings, method_settings, HPO_name#, save_results
    else:
        return "Could not retrieve parameters"



def get_folder (method_name, result_number) :
    # Get a list of all directories in the current directory
    directories = os.listdir(f"Results/{method_name}")

    # Filter out directories that start with 'result_'
    result_dirs = [d for d in directories if d.startswith(method_name)]

    # Extract the numeric part from the directory names
    numbers = [int(d.split('_')[1]) for d in result_dirs if d.split('_')[1].isdigit()]

    # Determine the next number in the sequence
    if numbers:
        if result_number == -1 :
            number = max(numbers)
            return f'Results/{method_name}/{method_name}_{number}'
        else :
            if result_number in numbers :
                return f'Results/{method_name}/{method_name}_{result_number}'
            else :
                return "No result with this number for this method"
    else:
        return "No result associated to this method"



def load_one_folder(method_name, result_number) :

    folder_path = get_folder (method_name, result_number)
    with open(folder_path+'/params_for_HPO.yaml', 'r') as file:
        params_for_HPO = file.read()

    benchmark_settings, method_settings, HPO_name = extract_benchmark_settings(params_for_HPO)
    HPO_settings = {"HPO_name": HPO_name}

    test_accs_path = folder_path+f'/test_accs_matrix.csv'
    with open(test_accs_path, 'r', newline='', encoding='utf-8') as f:
        test_accs_matrix = []
        reader = csv.reader(f)
        for row in reader:
            test_accs_matrix.append([float(element) for element in row])
        test_accs_matrix = np.array(test_accs_matrix)

    #best_params_path = folder_path+f'/best_params_{HPO_name}_{method_name}_{benchmark_name}_{difficulty}.csv'
    best_params_path = folder_path+f'/best_params.csv'
    with open(best_params_path, mode='r', newline='', encoding='utf-8') as file:
        best_params_list = []
        reader = csv.DictReader(file)
        for row in reader:
            best_params_list.append({key: float(value) for key, value in dict(row).items()})

    #val_accs_path = folder_path+f'/val_accs_matrix_{HPO_name}_{method_name}_{benchmark_name}_{difficulty}.csv'
    val_accs_path = folder_path+f'/val_accs_matrix.csv'
    with open(val_accs_path, 'r', newline='', encoding='utf-8') as f:
        val_accs_matrix = []
        reader = csv.reader(f)
        for row in reader:
            val_accs_matrix.append([float(element) for element in row])
        val_accs_matrix = np.array(val_accs_matrix)
    
    return test_accs_matrix, best_params_list, val_accs_matrix, HPO_settings, method_settings, benchmark_settings


def load(method_name) :
    path = f"Results/{method_name}"
    folders_names = os.listdir(path)
    result_numbers = [int(folder_name.split('_')[1]) for folder_name in folders_names]
    test_accs_matrix_list, best_params_list_list, val_accs_matrix_list = [], [], []
    HPO_settings_list, method_settings_list, benchmark_settings_list = [], [], []
    for result_number in result_numbers :
        test_accs_matrix, best_params_list, val_accs_matrix, HPO_settings, method_settings, benchmark_settings = load_one_folder(method_name, result_number)
        test_accs_matrix_list.append(test_accs_matrix)
        best_params_list_list.append(best_params_list)
        val_accs_matrix_list.append(val_accs_matrix)
        HPO_settings_list.append(HPO_settings)
        method_settings_list.append(method_settings)
        benchmark_settings_list.append(benchmark_settings)

    return test_accs_matrix_list, best_params_list_list, val_accs_matrix_list, HPO_settings_list, method_settings_list, benchmark_settings_list