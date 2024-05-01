import matplotlib.pyplot as plt


def plot_losses(train_loss_hist, val_loss_hist, comp_val_loss_hist=None):
    plt.plot(train_loss_hist,label="Optimized task : train",c="green")
    plt.plot(val_loss_hist,label="Optimized task : val",c="lime")
    
    if comp_val_loss_hist :
        plt.plot(comp_val_loss_hist,label="Side task : val",c="red")
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()


def plot_accs(train_acc_hist, val_acc_hist, comp_val_acc_hist=None):
    plt.plot(train_acc_hist,label="Optimized task : train",c="green")
    plt.plot(val_acc_hist,label="Optimized task : val",c="lime")
    
    if comp_val_acc_hist :
        plt.plot(comp_val_acc_hist,label="Side task : val",c="red")
    
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()