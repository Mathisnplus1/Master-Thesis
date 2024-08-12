import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib as mpl
from datetime import datetime


mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['legend.fontsize'] = 12



#################
###    HPO    ###
#################



def visualize_accs_matrix(test_accs_matrix_list, best_params_list_list, HPO_settings_list, method_settings_list, benchmark_settings_list, folder, savefig) :
    HPO_settings, method_settings, benchmark_settings = HPO_settings_list[0], method_settings_list[0], benchmark_settings_list[0]
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
    mean_test_accs_matrix = np.mean(test_accs_matrix_list, axis=0)
    num_tasks = len(mean_test_accs_matrix)
    plt.imshow(mean_test_accs_matrix, cmap='viridis', interpolation='nearest')
    plt.xticks(np.arange(num_tasks), np.arange(num_tasks))
    plt.yticks(np.arange(num_tasks), np.arange(num_tasks))
    plt.xlabel("Accuracy on task j...")
    plt.ylabel("...after training on task i")
    plt.colorbar()
    plt.tight_layout()
    
    # Save plot
    if savefig and folder is not None :
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(folder + f"/accs_matrix_{HPO_name}_{method_name}_from_{grow_from}_{benchmark_name}_{difficulty}_{current_time}.png")

    # Show plot
    plt.show()



def visualize_avg_acc_curve(test_accs_matrix_list, best_params_list_list, HPO_settings_list, method_settings_list, benchmark_settings_list, folder, savefig) :
    HPO_settings, method_settings, benchmark_settings = HPO_settings_list[0], method_settings_list[0], benchmark_settings_list[0]
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
    mean_test_accs_matrix = np.mean(test_accs_matrix_list, axis=0)
    num_tasks = len(mean_test_accs_matrix)
    mean_accs = np.array([np.array(mean_test_accs_matrix[i][:i+1]).sum() / (i+1) for i in range(num_tasks)])
    std_accs = np.array([np.array(mean_test_accs_matrix[i][:i+1]).std() for i in range(num_tasks)])
    plt.plot(range(num_tasks), mean_accs, color="black")
    plt.fill_between(range(num_tasks), mean_accs - std_accs, mean_accs + std_accs, color='black', alpha=0.2)
    plt.xlabel("Number of tasks trained")
    plt.ylabel("Test accuracy", fontsize=18)
    plt.xticks(np.arange(num_tasks), np.arange(num_tasks))
    plt.ylim(0, 100)
    plt.tight_layout()
    
    # Save plot
    if savefig and folder is not None :
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(folder + f"/avg_acc_curve_{HPO_name}_{method_name}_from_{grow_from}_{benchmark_name}_{difficulty}_{current_time}.png")

    # Show plot
    plt.show()



def format_float(values):
    if values[0] != int(values[0]) : # len(str(values[0])) > 5 :
        return [f"{float(value):.2e}" for value in values]
    else:
        return [str(int(value)) for value in values]
    


def visualize_best_params(test_accs_matrix_list, best_params_list_list, HPO_settings_list, method_settings_list, benchmark_settings_list, folder, savefig) :
    HPO_settings, method_settings, benchmark_settings = HPO_settings_list[0], method_settings_list[0], benchmark_settings_list[0]
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
    num_params = len(best_params_list_list[0][0])
    fig, axs = plt.subplots(nrows=1, ncols=num_params, figsize=(5*num_params, 5))
    plt.subplots_adjust(wspace=0.35)

    # Plot
    for ax, param_name in zip(axs, best_params_list_list[0][0].keys()) :
        mean_param_values = np.mean([[float(params[param_name]) for params in best_params_list] for best_params_list in best_params_list_list],axis=0)
        std_param_values = np.std([[float(params[param_name]) for params in best_params_list] for best_params_list in best_params_list_list],axis=0)
        ax.plot(mean_param_values, c="black")
        ax.fill_between(range(len(mean_param_values)), mean_param_values - std_param_values, mean_param_values + std_param_values, color='black', alpha=0.2)
        ax.set_xticks(range(len(mean_param_values)))
        #ax.set_yticks(param_values, format_float(param_values))
        ax.set_ylabel(f"Best {param_name}")
        ax.set_xlabel("Task index")
    plt.tight_layout()

    # Save plot
    if savefig and folder is not None :
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(folder + f"/best_params_{HPO_name}_{method_name}_from_{grow_from}_{benchmark_name}_{difficulty}_{current_time}.png")

    # Show plot
    plt.show()



def visualize_HPO(test_accs_matrix_list, best_params_list_list, 
                  visualization_settings, HPO_settings_list, 
                  method_settings_list, benchmark_settings_list, 
                  method_name=None) :
    folder = None
    if method_name :
        folder = f"Results/{method_name}/"
    
    functions_list = [visualize_accs_matrix, visualize_avg_acc_curve, visualize_best_params]
    
    for function in functions_list :
        if visualization_settings[function.__name__] :
            function(test_accs_matrix_list, best_params_list_list, HPO_settings_list, method_settings_list, benchmark_settings_list, folder, visualization_settings["savefig"])



##################
### VALIDATION ###
##################



def visualize_val_accs_matrix(combined_val_accs_matrix_list, HPO_settings_list, method_settings_list, benchmark_settings_list, folder, savefig=False):
    HPO_settings, method_settings, benchmark_settings = HPO_settings_list[0], method_settings_list[0], benchmark_settings_list[0]
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
        num_tasks = benchmark_settings["num_tasks"]
    except ValueError :
        print("One or more of the required settings to visualize are missing. Please check the benchmark_settings.")

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    im = axs.imshow(np.mean(combined_val_accs_matrix_list, axis=0), cmap='viridis', interpolation='nearest')
    axs.set_yticks(np.arange(1+num_val_benchmarks), ["HPO"]+[f"Val {i}" for i in range(1,num_val_benchmarks+1)])
    axs.set_xticks(np.arange(num_tasks), np.arange(num_tasks))
    axs.set_xlabel("Task index")
    plt.colorbar(im)
    plt.tight_layout()

    # Save plot
    if savefig and folder is not None :
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(folder + f"/val_accs_matrix_{HPO_name}_{method_name}_from_{grow_from}_{benchmark_name}_{difficulty}_{current_time}.png")

    # Show plot
    plt.show()



def visualize_accuracy_through_benchmarks (combined_val_accs_matrix_list, HPO_settings_list, method_settings_list, benchmark_settings_list, folder, savefig=False) :
    HPO_settings, method_settings, benchmark_settings = HPO_settings_list[0], method_settings_list[0], benchmark_settings_list[0]
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
    
    # Create subplots
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8, 5), width_ratios= [6, 2])
    plt.subplots_adjust(wspace=0.35)

    # Calculate the mean and standard deviation along the axis of experiments
    mean_results = np.mean(np.mean(combined_val_accs_matrix_list, axis=0)[1:], axis=0)
    std_dev_results = np.std(np.mean(combined_val_accs_matrix_list, axis=0)[1:], axis=0)

    # X-axis values
    x = np.arange(combined_val_accs_matrix_list[0].shape[1])

    # Plotting the mean results and out of HPO result
    axs[0].plot(x, np.mean(combined_val_accs_matrix_list, axis=0)[0], label="HPO benchmark", color='black')
    axs[0].plot(x, mean_results, label=f'Mean val benchmarks', color='black', linestyle='--')

    # Shaded area for standard deviation
    axs[0].fill_between(x, mean_results - std_dev_results, mean_results + std_dev_results, color='black', alpha=0.2)

    # Adding labels and title
    axs[0].set_xlabel('Task index')
    axs[0].set_xticks(x)
    axs[0].set_ylabel('Test Accuracy')
    axs[0].legend()

    # Second plot
    mean = np.mean(combined_val_accs_matrix_list, axis=0)[1:].mean(axis=0).mean()
    std = np.mean(combined_val_accs_matrix_list, axis=0)[1:].mean(axis=0).std()
    delta = 0.03*(axs[1]._get_view()["ylim"][1] - axs[1]._get_view()["ylim"][0])# Pour taille des crochets
    # + Std
    axs[1].plot([0.85,1.15], 2*[mean + std], color=(0.6,0.6,0.6), linewidth=2)
    axs[1].plot([0.85,0.85], [mean + std -delta, mean + std], color=(0.6,0.6,0.6), linewidth=2)
    axs[1].plot([1.15,1.15], [mean + std -delta, mean + std], color=(0.6,0.6,0.6), linewidth=2)
    # Middle
    axs[1].plot([1,1], [mean - std, mean + std], color=(0.6,0.6,0.6), linewidth=2)
    axs[1].fill_between([0.855,1.145], mean - std, mean + std, color="black", alpha=0.2)
    # - Std
    axs[1].plot([0.85,1.15], 2*[mean - std], color=(0.6,0.6,0.6), linewidth=2)
    axs[1].plot([0.85,0.85], [mean - std, mean - std +delta], color=(0.6,0.6,0.6), linewidth=2)
    axs[1].plot([1.15,1.15], [mean - std, mean - std +delta], color=(0.6,0.6,0.6), linewidth=2)
    # Means
    axs[1].plot([0.9,1.1], 2*[np.mean(combined_val_accs_matrix_list, axis=0)[0].mean()], color='black')
    axs[1].plot([0.9,1.1], 2*[mean], color='black', linestyle='--')

    

    axs[1].set_xlim(0.7, 1.3)

    # Adding labels and title
    axs[1].set_xticks(ticks=[1], labels=["Mean"])
    #axs[1].tick_params(labelsize=18)

    plt.tight_layout()

    # Save plot
    if savefig and folder is not None :
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(folder)
        plt.savefig(folder + f"/accuracy_through_benchmarks_{HPO_name}_{method_name}_from_{grow_from}_{benchmark_name}_{difficulty}_{current_time}.png")

    # Show plot
    plt.show()
   


def visualize_validation(val_accs_matrix_list, test_accs_matrix_list,
                         visualization_settings, HPO_settings_list,
                         method_settings_list, benchmark_settings_list,
                         method_name=None) :
    folder = None
    if method_name :
        folder = f"Results/{method_name}/"
    combined_val_accs_matrix_list = []
    for test_accs_matrix, val_accs_matrix in zip(test_accs_matrix_list, val_accs_matrix_list) :
        combined_val_accs_matrix = np.concatenate((np.reshape(test_accs_matrix[-1], (1,test_accs_matrix.shape[1])), val_accs_matrix), axis=0)
        combined_val_accs_matrix_list.append(combined_val_accs_matrix)
    
    functions_list = [visualize_val_accs_matrix, visualize_accuracy_through_benchmarks]
    for function in functions_list :
        if visualization_settings[function.__name__] :
            function(combined_val_accs_matrix_list, HPO_settings_list, method_settings_list, benchmark_settings_list, folder, visualization_settings["savefig"])
