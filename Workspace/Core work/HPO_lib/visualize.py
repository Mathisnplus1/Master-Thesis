import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from datetime import datetime



#################
###    HPO    ###
#################



def visualize_accs_matrix(test_accs_matrix, best_params_list, HPO_settings, method_settings, benchmark_settings, savefig) :
    # Check if all required settings are present
    try :
        HPO_name = HPO_settings["HPO_name"]
    except ValueError :
        print("One or more of the required settings to visualize are missing. Please check the HPO_settings.")
    
    try :
        method_name = method_settings["method_name"]
        grow_from = method_settings["grow_from"]
    except ValueError :
        print("One or more of the required settings to visualize are missing. Please check the method_settings.")
    
    try :
        benchmark_name = benchmark_settings["benchmark_name"]
        difficulty = benchmark_settings["difficulty"]
    except ValueError :
        print("One or more of the required settings to visualize are missing. Please check the benchmark_settings.")

    # Plot
    num_tasks = len(test_accs_matrix)
    plt.imshow(test_accs_matrix, cmap='viridis', interpolation='nearest')
    plt.yticks(np.arange(num_tasks), np.arange(num_tasks))
    plt.xlabel("Accuracy on task j...")
    plt.ylabel("...after training on task i")
    plt.colorbar()
    
    # Save plot
    if savefig:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"Results/accs_matrix_{HPO_name}_{method_name}_from_{grow_from}_{benchmark_name}_{difficulty}_{current_time}.png")

    # Show plot
    plt.show()



def visualize_avg_acc_curve(test_accs_matrix, best_params_list, HPO_settings, method_settings, benchmark_settings, savefig) :
    # Check if all required settings are present
    try :
        HPO_name = HPO_settings["HPO_name"]
    except ValueError :
        print("One or more of the required settings to visualize are missing. Please check the HPO_settings.")
    
    try :
        method_name = method_settings["method_name"]
        grow_from = method_settings["grow_from"]
    except ValueError :
        print("One or more of the required settings to visualize are missing. Please check the method_settings.")
    
    try :
        benchmark_name = benchmark_settings["benchmark_name"]
        difficulty = benchmark_settings["difficulty"]
    except ValueError :
        print("One or more of the required settings to visualize are missing. Please check the benchmark_settings.")

    # Plot
    num_tasks = len(test_accs_matrix)
    mean_accs = [np.array(test_accs_matrix[i][:i+1]).sum() / (i+1) for i in range(num_tasks)]
    plt.plot(range(num_tasks), mean_accs, color="black")
    plt.xlabel("Number of tasks trained")
    plt.ylabel("Average test accuracy for tasks trained so far")
    plt.ylim(0, 100)
    
    # Save plot
    if savefig:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"Results/avg_acc_curve_{HPO_name}_{method_name}_from_{grow_from}_{benchmark_name}_{difficulty}_{current_time}.png")

    # Show plot
    plt.show()



def visualize_best_params(test_accs_matrix, best_params_list, HPO_settings, method_settings, benchmark_settings, savefig) :
    # Check if all required settings are present
    try :
        HPO_name = HPO_settings["HPO_name"]
    except ValueError :
        print("One or more of the required settings to visualize are missing. Please check the HPO_settings.")
    
    try :
        method_name = method_settings["method_name"]
        grow_from = method_settings["grow_from"]
    except ValueError :
        print("One or more of the required settings to visualize are missing. Please check the method_settings.")
    
    try :
        benchmark_name = benchmark_settings["benchmark_name"]
        difficulty = benchmark_settings["difficulty"]
    except ValueError :
        print("One or more of the required settings to visualize are missing. Please check the benchmark_settings.")

    # Create a subplot for each param
    num_params = len(best_params_list[0])
    fig, axs = plt.subplots(nrows=1, ncols=num_params, figsize=(5*num_params, 5))
    
    # Plot
    for ax, param_name in zip(axs, best_params_list[0].keys()) :
        param_values = [params[param_name] for params in best_params_list]
        ax.plot(param_values)
        ax.set_xticks(range(len(param_values)))
        ax.set_xlabel("Task index")
        ax.set_title(f"Best {param_name}")

    # Save plot
    if savefig:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"Results/best_params_{HPO_name}_{method_name}_from_{grow_from}_{benchmark_name}_{difficulty}_{current_time}.png")

    # Show plot
    plt.show()



def visualize_HPO(test_accs_matrix, best_params_list, visualization_settings, HPO_settings, method_settings, benchmark_settings) :
    if HPO_settings["HPO_name"] == "greedy_HPO" :
        functions_list = [visualize_accs_matrix, visualize_avg_acc_curve, visualize_best_params]
    if HPO_settings["HPO_name"] == "cheated_HPO" :
        functions_list = [visualize_accs_matrix, visualize_avg_acc_curve]
    for function in functions_list :
        if visualization_settings[function.__name__] :
            function(test_accs_matrix, best_params_list, HPO_settings, method_settings, benchmark_settings, visualization_settings["savefig"])



##################
### VALIDATION ###
##################



def visualize_val_accs_matrix(combined_val_accs_matrix, HPO_settings, method_settings, benchmark_settings, savefig=False):
    # Check if all required settings are present
    try :
        HPO_name = HPO_settings["HPO_name"]
    except ValueError :
        print("One or more of the required settings to visualize are missing. Please check the HPO_settings.")
    
    try :
        method_name = method_settings["method_name"]
        grow_from = method_settings["grow_from"]
    except ValueError :
        print("One or more of the required settings to visualize are missing. Please check the method_settings.")
    
    try :
        benchmark_name = benchmark_settings["benchmark_name"]
        num_val_benchmarks = benchmark_settings["num_val_benchmarks"]
        difficulty = benchmark_settings["difficulty"]
    except ValueError :
        print("One or more of the required settings to visualize are missing. Please check the benchmark_settings.")
    
    plt.imshow(combined_val_accs_matrix, cmap='viridis', interpolation='nearest')
    plt.yticks(np.arange(2+num_val_benchmarks), ["HPO's benchmark", "HPO's benchmark, reshuffled"]+[f"Val benchmark {i}" for i in range(1,num_val_benchmarks+1)])
    plt.xlabel("Task index")
    plt.colorbar()

    # Save plot
    if savefig:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"Results/val_accs_matrix_{HPO_name}_{method_name}_from_{grow_from}_{benchmark_name}_{difficulty}_{current_time}.png")

    # Show plot
    plt.show()



def visualize_accuracy_through_benchmarks (combined_val_accs_matrix, HPO_settings, method_settings, benchmark_settings, savefig=False) :
    # Check if all required settings are present
    try :
        HPO_name = HPO_settings["HPO_name"]
    except ValueError :
        print("One or more of the required settings to visualize are missing. Please check the HPO_settings.")
    
    try :
        method_name = method_settings["method_name"]
        grow_from = method_settings["grow_from"]
    except ValueError :
        print("One or more of the required settings to visualize are missing. Please check the method_settings.")
    
    try :
        benchmark_name = benchmark_settings["benchmark_name"]
        num_val_benchmarks = benchmark_settings["num_val_benchmarks"]
        difficulty = benchmark_settings["difficulty"]
    except ValueError :
        print("One or more of the required settings to visualize are missing. Please check the benchmark_settings.")
    
    # Calculate the mean and standard deviation along the axis of experiments
    mean_results = np.mean(combined_val_accs_matrix[2:], axis=0)
    std_dev_results = np.std(combined_val_accs_matrix[2:], axis=0)

    # X-axis values
    x = np.arange(combined_val_accs_matrix.shape[1])

    # Plotting the mean results and out of HPO result
    plt.plot(x, combined_val_accs_matrix[0], label="HPO's benchmark", color='r')
    plt.plot(x, combined_val_accs_matrix[1], label="HPO's benchmark, reshuffled", color='g')
    plt.plot(x, mean_results, label=f'Mean through {num_val_benchmarks} other benchmarks', color='b')

    # Shaded area for standard deviation
    plt.fill_between(x, mean_results - std_dev_results, mean_results + std_dev_results, color='b', alpha=0.2)

    # Adding labels and title
    plt.xlabel('Task index')
    plt.xticks(x)
    plt.ylabel('Test Accuracy')
    plt.legend()

    # Save plot
    if savefig:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"Results/accuracy_through_benchmarks_{HPO_name}_{method_name}_from_{grow_from}_{benchmark_name}_{difficulty}_{current_time}.png")

    # Show plot
    plt.show()



def visualize_violin(combined_val_accs_matrix, HPO_settings, method_settings, benchmark_settings, savefig=False) :
    # Check if all required settings are present
    try :
        HPO_name = HPO_settings["HPO_name"]
    except ValueError :
        print("One or more of the required settings to visualize are missing. Please check the HPO_settings.")
    
    try :
        method_name = method_settings["method_name"]
        grow_from = method_settings["grow_from"]
    except ValueError :
        print("One or more of the required settings to visualize are missing. Please check the method_settings.")
    
    try :
        benchmark_name = benchmark_settings["benchmark_name"]
        difficulty = benchmark_settings["difficulty"]
    except ValueError :
        print("One or more of the required settings to visualize are missing. Please check the benchmark_settings.")
    
    # Resizing the plot
    plt.figure(figsize=(2, 5))

    # Creating the violin plot
    violin = plt.violinplot(combined_val_accs_matrix[2:].mean(axis=0), showmeans=True, showextrema=False)
    violin['bodies'][0].set_facecolor("b")
    violin['bodies'][0].set_alpha(0.2)
    violin['cmeans'].set_color('b')
    plt.scatter(1, combined_val_accs_matrix[0].mean(), color='r')
    plt.scatter(1, combined_val_accs_matrix[1].mean(), color='g')

    # Adding labels and title
    plt.xticks([])
    plt.ylabel('Test Accuracy')

    # Save plot
    if savefig:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"Results/violin_{HPO_name}_{method_name}_from_{grow_from}_{benchmark_name}_{difficulty}_{current_time}.png")

    # Show plot
    plt.show()



def visualize_validation(val_accs_matrix, test_accs_matrix, visualization_settings, HPO_settings, method_settings, benchmark_settings) :
    combined_val_accs_matrix = np.concatenate((np.reshape(test_accs_matrix[-1], (1,test_accs_matrix.shape[1])), val_accs_matrix), axis=0)
    functions_list = [visualize_val_accs_matrix, visualize_accuracy_through_benchmarks, visualize_violin]
    for function in functions_list :
        if visualization_settings[function.__name__] :
            function(combined_val_accs_matrix, HPO_settings, method_settings, benchmark_settings, visualization_settings["savefig"])
