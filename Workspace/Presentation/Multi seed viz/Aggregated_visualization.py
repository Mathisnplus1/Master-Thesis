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
mpl.rcParams['legend.fontsize'] = 14




def visualize_avg_acc_curve(test_accs_matrix_list, label) :
    # Plot
    mean_test_accs_matrix = np.mean(test_accs_matrix_list, axis=0)
    num_tasks = len(mean_test_accs_matrix)
    mean_accs = np.array([np.array(mean_test_accs_matrix[i][:i+1]).sum() / (i+1) for i in range(num_tasks)])
    std_accs = np.array([np.array(mean_test_accs_matrix[i][:i+1]).std() for i in range(num_tasks)])
    plt.plot(range(num_tasks), mean_accs, label=label)
    plt.fill_between(range(num_tasks), mean_accs - std_accs, mean_accs + std_accs, alpha=0.2)



def visualize_agg_avg_acc_curve(dict_test_accs_matrix_list, folder, savefig) :
    for label, test_accs_matrix_list in dict_test_accs_matrix_list.items() :
        visualize_avg_acc_curve(test_accs_matrix_list, label)

    plt.xlabel("Number of tasks trained")
    plt.ylabel("Test accuracy", fontsize=18)
    plt.xticks(np.arange(10), np.arange(10))
    plt.ylim(60, 100)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    if savefig and folder is not None :
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(folder + f"/agg_avg_acc_curve_{current_time}.png")

    # Show plot
    plt.show()


def visualize_litterature(Ours, Regs, Reps, savefig=False) :
    # Create the plot
    plt.figure(figsize=(5, 6))  # Set the figure size to be wide and short
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan", "gold", "lightsteelblue", "springgreen", "navy", "salmon", "darkorchid", "khaki", "peru"]
    colors.reverse()

    # Our results
    i=1
    for label, l in Ours.items() :
        mean, std = np.mean(l), np.std(l)
        c = colors.pop()
        plt.scatter((i//2+i%2)*0.1*(-1)**i, mean, label=label, color=c)
        plt.errorbar((i//2+i%2)*0.1*(-1)**i, mean, yerr=std, fmt='o', color=c) 
        i+=1

    # Regularization results
    for label, l in Regs.items() :
        mean, std = np.mean(l), np.std(l)
        plt.scatter(0, mean, label=label, color=colors.pop())
        #plt.errorbar(0, mean, yerr=std, fmt='o')
    
    # Replay results
    for label, l in Reps.items() :
        mean, std = np.mean(l), np.std(l)
        plt.scatter(0, mean, label=label, color=colors.pop())
        #plt.errorbar(0, mean, yerr=std, fmt='o')

    # Customize the plot
    plt.xticks([])
    plt.xlim([-0.3, 0.9])
    #plt.xlabel('Value')
    plt.ylabel('Test accuracy')

    plt.legend(loc="upper right")
    plt.tight_layout()

    # Save
    if savefig : #savefig and folder is not None :
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"lit_{current_time}.png")

    plt.show()


def visualize_grouped_litterature(Ours, Regs, Reps, savefig=False) :
    # Create the plot
    plt.figure(figsize=(5, 6))  # Set the figure size to be wide and short

    # Our results
    i = 1
    for label, l in Ours.items() :
        mean, std = np.mean(l), np.std(l)
        if label == "GroHess" :
            plt.scatter((i//2+i%2)*0.1*(-1)**i, mean, label=label, color='red')
            plt.errorbar((i//2+i%2)*0.1*(-1)**i, mean, yerr=std, fmt='o', color='red')
        elif i == 2 :
            plt.scatter((i//2+i%2)*0.1*(-1)**i, mean, label="Baselines", color='tomato')
            plt.errorbar((i//2+i%2)*0.1*(-1)**i, mean, yerr=std, fmt='o', color='tomato')
        else :
            plt.scatter((i//2+i%2)*0.1*(-1)**i, mean, color='tomato')
            plt.errorbar((i//2+i%2)*0.1*(-1)**i, mean, yerr=std, fmt='o', color='tomato')
        i+=1

    # Regularization results
    i = 1
    for label, l in Regs.items() :
        mean, std = np.mean(l), np.std(l)
        if i == 1 :
            plt.scatter(0, mean, label="Regularization", color='gold')
        else :
            plt.scatter(0, mean, color='gold')
        i+=1
    
    # Replay results
    i = 1
    for label, l in Reps.items() :
        mean, std = np.mean(l), np.std(l)
        if i == 1 :
            plt.scatter(0, mean, label="Rehearsal", color='limegreen')
        else :
            plt.scatter(0, mean, color='limegreen')
        i+=1

    # Customize the plot
    plt.xticks([])
    plt.xlim([-0.4, 0.8])
    #plt.xlabel('Value')
    plt.ylabel('Test accuracy')

    plt.legend(loc="upper right")
    plt.tight_layout()

    # Save
    if savefig : #savefig and folder is not None :
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"lit_{current_time}.png")

    plt.show()