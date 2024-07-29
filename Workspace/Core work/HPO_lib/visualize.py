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



def visualize_accs_matrix(test_accs_matrix, best_params_list, HPO_settings, method_settings, benchmark_settings, folder, savefig) :
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



def visualize_avg_acc_curve(test_accs_matrix, best_params_list, HPO_settings, method_settings, benchmark_settings, folder, savefig) :
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
    


def visualize_best_params(test_accs_matrix, best_params_list, HPO_settings, method_settings, benchmark_settings, folder, savefig) :
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
    plt.subplots_adjust(wspace=0.35)

    # Plot
    for ax, param_name in zip(axs, best_params_list[0].keys()) :
        param_values = [float(params[param_name]) for params in best_params_list]
        ax.plot(param_values)
        ax.set_xticks(range(len(param_values)))
        ax.set_yticks(param_values, format_float(param_values))
        ax.set_ylabel(f"Best {param_name}")
        ax.set_xlabel("Task index")
    plt.tight_layout()

    # Save plot
    if savefig and folder is not None :
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(folder + f"/best_params_{HPO_name}_{method_name}_from_{grow_from}_{benchmark_name}_{difficulty}_{current_time}.png")

    # Show plot
    plt.show()



def visualize_HPO(test_accs_matrix, best_params_list, visualization_settings, HPO_settings, method_settings, benchmark_settings, folder=None) :
    if HPO_settings["HPO_name"] == "greedy_HPO" :
        functions_list = [visualize_accs_matrix, visualize_avg_acc_curve, visualize_best_params]
    if HPO_settings["HPO_name"] == "cheated_HPO" :
        functions_list = [visualize_accs_matrix, visualize_avg_acc_curve]
    for function in functions_list :
        if visualization_settings[function.__name__] :
            function(test_accs_matrix, best_params_list, HPO_settings, method_settings, benchmark_settings, folder, visualization_settings["savefig"])



##################
### VALIDATION ###
##################



def visualize_val_accs_matrix(combined_val_accs_matrix, HPO_settings, method_settings, benchmark_settings, folder, savefig=False):
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
    im = axs.imshow(combined_val_accs_matrix, cmap='viridis', interpolation='nearest')
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



def visualize_accuracy_through_benchmarks (combined_val_accs_matrix, HPO_settings, method_settings, benchmark_settings, folder, savefig=False) :
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
    mean_results = np.mean(combined_val_accs_matrix[1:], axis=0)
    std_dev_results = np.std(combined_val_accs_matrix[1:], axis=0)

    # X-axis values
    x = np.arange(combined_val_accs_matrix.shape[1])

    # Plotting the mean results and out of HPO result
    axs[0].plot(x, combined_val_accs_matrix[0], label="HPO benchmark", color='r')
    axs[0].plot(x, mean_results, label=f'Mean val benchmarks', color='b')

    # Shaded area for standard deviation
    axs[0].fill_between(x, mean_results - std_dev_results, mean_results + std_dev_results, color='b', alpha=0.2)

    # Adding labels and title
    axs[0].set_xlabel('Task index')
    axs[0].set_xticks(x)
    axs[0].set_ylabel('Test Accuracy')
    axs[0].legend()

    # Creating the violin plot
    violin = plt.violinplot(combined_val_accs_matrix[1:].mean(axis=0), widths=0.4, showmeans=True, showextrema=False)
    violin['bodies'][0].set_facecolor("b")
    violin['bodies'][0].set_alpha(0.2)
    violin['cmeans'].set_color('b')
    #violin['cmeans'].set_linewidth(10)
    axs[1].plot([0.9,1.1], 2*[combined_val_accs_matrix[0].mean()], color='r')

    # Adding labels and title
    axs[1].set_xticks(ticks=[1], labels=["Mean"])
    #axs[1].tick_params(labelsize=18)

    plt.tight_layout()

    # Save plot
    if savefig and folder is not None :
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(folder + f"Results/accuracy_through_benchmarks_{HPO_name}_{method_name}_from_{grow_from}_{benchmark_name}_{difficulty}_{current_time}.png")

    # Show plot
    plt.show()
   


def visualize_validation(val_accs_matrix, test_accs_matrix, visualization_settings, HPO_settings, method_settings, benchmark_settings, folder=None) :
    combined_val_accs_matrix = np.concatenate((np.reshape(test_accs_matrix[-1], (1,test_accs_matrix.shape[1])), val_accs_matrix), axis=0)
    functions_list = [visualize_val_accs_matrix, visualize_accuracy_through_benchmarks]
    for function in functions_list :
        if visualization_settings[function.__name__] :
            function(combined_val_accs_matrix, HPO_settings, method_settings, benchmark_settings, folder, visualization_settings["savefig"])
