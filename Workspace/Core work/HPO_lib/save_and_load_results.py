import pickle

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

        with open(f'Results/test_accs_matrix_{HPO_name}_{method_name}_{benchmark_name}_{difficulty}.pkl', 'wb') as f:
            pickle.dump(test_accs_matrix, f)
        
        with open(f'Results/best_params{HPO_name}_{method_name}_{benchmark_name}_{difficulty}.pkl', 'wb') as f:
            pickle.dump(best_params_list, f)
        
        with open(f'Results/val_accs_matrix_{HPO_name}_{method_name}_{benchmark_name}_{difficulty}.pkl', 'wb') as f:
            pickle.dump(val_accs_matrix, f)



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

    with open (f'Results/test_accs_matrix_{HPO_name}_{method_name}_{benchmark_name}_{difficulty}.pkl', 'r') as f :
        test_accs_matrix = pickle.load(f)
    with open (f'Results/best_params{HPO_name}_{method_name}_{benchmark_name}_{difficulty}.pkl', 'r') as f :
        best_params_list = pickle.load(f)
    with open (f'Results/val_accs_matrix_{HPO_name}_{method_name}_{benchmark_name}_{difficulty}.pkl', 'r') as f :
        val_accs_matrix = pickle.load(f)
    
    return test_accs_matrix, best_params_list, val_accs_matrix