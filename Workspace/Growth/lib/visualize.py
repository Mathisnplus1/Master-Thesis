import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np



def get_path (num_hidden, num_hidden_target, growth_schedule) :
    x, y = num_hidden, num_hidden
    x_path, y_path = [x], [y]
    for layer_name, num_neurons in growth_schedule :
        if layer_name == "fc1" :
            x += num_neurons
        elif layer_name == "fc2" :
            y += num_neurons
        x_path.append(x)
        y_path.append(y)
        
    return x_path, y_path


def get_color (c_scale, min_acc, max_acc, test_acc) :
    r = (test_acc-min_acc) / (max_acc-min_acc)
    color_index = int(r * (len(c_scale)-1))
    return c_scale[color_index]


def visualize_pathes (num_hidden, num_hidden_target, growth_schedules, 
                      test_accs, savefig=False) :
    
    test_accs = test_accs.tolist()
    
    fig, ax = plt.subplots(figsize=(5,5))
    plt.scatter(num_hidden,num_hidden, color="black")
    plt.text(num_hidden, num_hidden-5, "root", fontsize=10, 
             color="black")
    plt.scatter(num_hidden_target,num_hidden_target, color="black")
    plt.text(num_hidden_target, num_hidden_target+2, "target", fontsize=10, 
             color="black")

    
    min_acc, max_acc = min(test_accs), max(test_accs)
    c_scale = cm.viridis(np.linspace(0, 1, 60))
    
    for i, (growth_schedule, test_acc) in enumerate(zip(growth_schedules, test_accs)) :
        x_path, y_path = get_path(num_hidden, num_hidden_target, growth_schedule)
        color = get_color(c_scale, min_acc, max_acc, test_acc)
        ax.plot(x_path, y_path, c=color, label=str(test_acc))
    
    ax.set_xlim(0,num_hidden_target+15)
    ax.set_ylim(0,num_hidden_target+15)
    ax.set_xlabel("$n_{1}$")
    ax.set_ylabel("$n_{2}$")
    
    #plt.legend()
    
    # Create colorbar axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Plot Viridis color scale
    cbar = plt.colorbar(cm.ScalarMappable(cmap='viridis', 
                                          norm=plt.Normalize(vmin=min_acc,
                                                             vmax=max_acc)),
                        cax=cax)
    cbar.set_label('Test accuracy')
    
    ax.set_title("Growth pathes and their average test accuracy across 5 trials")
    
    if savefig :
        plt.savefig("MNIST_pathes.png")
    
    plt.show()




def visualize_statistical_reliability (test_accs_repeted, test_accs, test_acc_roots, test_acc_targets,
                                       free_lim=True, savefig=False) :
    # Create figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [4, 1, 1]})
    fig.suptitle("Average test accuracy across 5 trials, with standard deviation")
    axs[0].set_ylabel("Test accuracy")
    
    # Paths
    test_accs_means = np.mean(test_accs_repeted, 0)
    test_accs_stds = np.std(test_accs_repeted, 0)
    
    min_acc, max_acc = min(test_accs), max(test_accs)
    c_scale = cm.viridis(np.linspace(0, 1, 60))
    colors = [get_color(c_scale, min_acc, max_acc, test_acc) for test_acc in test_accs]
    axs[0].errorbar(range(1,test_accs.shape[0]+1), test_accs_means, yerr=test_accs_stds, fmt="none", ecolor=colors)
    axs[0].scatter(range(1,test_accs.shape[0]+1), test_accs_means, marker='.', s=100, color=colors)
    axs[0].set_xlabel("On each path")
    
    # Root
    axs[1].errorbar(1, np.mean(test_acc_roots), yerr=np.std(test_acc_roots), color="black")
    axs[1].scatter(1, np.mean(test_acc_roots), marker='.', s=100, color="black")
    axs[1].set_xlabel("On the root model")
    
    # Target
    axs[2].errorbar(1, np.mean(test_acc_targets), yerr=np.std(test_acc_targets), color="black")
    axs[2].scatter(1, np.mean(test_acc_targets), marker='.', s=100, color="black")
    axs[2].set_xlabel("On the target model")
    
    # Set name tag
    free_lim_tag = ""
    if not free_lim :
        # Edit name tag
        free_lim_tag = "_free_lim"
        
        # Set reference for y lim
        root_std = np.std(test_acc_roots)
        
        # Set y lims
        axs[0].set_ylim(test_accs_means.mean() - root_std-2,test_accs_means.mean() + root_std-2)
        axs[1].set_ylim(np.mean(test_acc_roots) - root_std-2,np.mean(test_acc_roots) + root_std+2)
        axs[2].set_ylim(np.mean(test_acc_targets) - root_std-2,np.mean(test_acc_targets) + root_std+2)
    
    if savefig :
        plt.savefig("MNIST_statistical_reliability"+free_lim_tag+".png")

    plt.show()



def visualize_box_plot (test_accs, test_acc_root, test_acc_target, savefig=False):
    
    test_accs = test_accs.tolist()
    
    test_accs = test_accs+[test_acc_root, test_acc_target]
    
    min_acc, max_acc = min(test_accs), max(test_accs)
    c_scale = cm.viridis(np.linspace(0, 1, 60))

    # Plotting
    fig, ax = plt.subplots(figsize=(3, 5))
    
    plt.boxplot([test_accs], patch_artist=True, widths=0.5, showfliers=False,
                boxprops=dict(facecolor='none'),
                medianprops=dict(color='black'))
    
    xs = np.random.normal(1, 0.04, size=len(test_accs[:-2]))  # Adding jitter for better visibility
    for x, test_acc in zip(xs, test_accs[:-2]) :
        plt.plot(x, test_acc, 'o', markersize=5, c=get_color(c_scale, min_acc, max_acc, test_acc))
    
    plt.plot(1, test_acc_root, 'o', markersize=10, c=get_color(c_scale, min_acc, max_acc, test_acc_root))
    plt.text(1.1, test_acc_root, "root", fontsize=10, color=get_color(c_scale, min_acc, max_acc, test_acc_root))
    plt.plot(1, test_acc_target, 'o', markersize=10, c=get_color(c_scale, min_acc, max_acc, test_acc_target))
    plt.text(1.1, test_acc_target, "target", fontsize=10, color=get_color(c_scale, min_acc, max_acc, test_acc_target))
    

    plt.ylabel('Test accuracies')
    plt.ylim(0,100)
    plt.xticks([])
    
    plt.grid(True)
    
    plt.title("Average test accuracy across 5 trials of\npathes, root model and target model",
              fontsize=10)
    
    # Create colorbar axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Plot Viridis color scale
    cbar = plt.colorbar(cm.ScalarMappable(cmap='viridis', 
                                          norm=plt.Normalize(vmin=min_acc,
                                                             vmax=max_acc)),
                        cax=cax)
    
    if savefig :
        plt.savefig("MNIST_box_plot.png")
    
    plt.show()