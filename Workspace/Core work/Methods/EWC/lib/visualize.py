import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def vizualize_loss_hists(loss_hist_list, growth_indices_list) :
    for i, (loss_hist, growth_indices) in enumerate(zip(loss_hist_list, growth_indices_list)) :
        plt.plot(loss_hist)
        plt.vlines(x=growth_indices, ymin=min(loss_hist), ymax=max(loss_hist),color='red', linestyle='--')
        plt.title("Loss history while training on task " + str(i))
        plt.xlabel("Batch number")
        plt.ylabel("Loss")
        plt.show()


def visualize_accs_matrix(test_accs_matrix, savefig=False):
    num_tasks = len(test_accs_matrix)
    plt.imshow(test_accs_matrix, cmap='viridis', interpolation='nearest')
    plt.yticks(np.arange(num_tasks), np.arange(num_tasks))
    plt.xlabel("Accuracy on task j...")
    plt.ylabel("...after training on task i")
    plt.title(f"None")
    plt.colorbar()
    if savefig :
        plt.savefig(f"results/None.png")
    plt.show()


def visualize_avg_acc_curve(test_accs_matrix, savefig=False):
    num_tasks = len(test_accs_matrix)
    mean_accs = [np.array(test_accs_matrix[i][:i+1]).sum() / (i+1) for i in range(num_tasks)]
    plt.plot(range(num_tasks), mean_accs, color="black")

    plt.xlabel("Number of tasks trained")
    plt.ylabel("Average accuracy for tasks trained so far")
    plt.ylim(0, 100)
    plt.title("None")
    
    if savefig:
        plt.savefig(f"results/None.png")
    
    plt.show()