import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def visualize_accs_matrix(test_accs_matrix, savefig=False):
    plt.imshow(test_accs_matrix, cmap='viridis', interpolation='nearest')
    plt.yticks(np.arange(10), np.arange(10))
    plt.xlabel("Accuracy on digit j...")
    plt.ylabel("...after training on digit i")
    plt.title(f"None")
    plt.colorbar()
    if savefig :
        plt.savefig(f"None.png")
    plt.show()


def visualize_avg_acc_curve(test_accs_matrix, savefig=False):
    mean_accs = [np.array(test_accs_matrix[i][:i+1]).sum() / (i+1) for i in range(10)]
    plt.plot(range(10), mean_accs, color="black")

    plt.xlabel("Number of classes trained")
    plt.ylabel("Average accuracy for classes trained so far")
    plt.ylim(0, 100)
    plt.title("None")
    
    if savefig:
        plt.savefig(f"results/None.png")
    
    plt.show()