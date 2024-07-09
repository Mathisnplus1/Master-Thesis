from lib.models import ANN
from abstract_torch import get_optimizer, get_loss
from avalanche.models import SimpleCNN
import numpy as np

from avalanche.training.supervised import LwF


def initialize_model(method_settings, global_seed) :
    try :
        num_inputs = method_settings["num_inputs"]
        num_hidden_root = method_settings["num_hidden_root"]
        num_outputs = method_settings["num_outputs"]
    except ValueError:
        print("One or more of the required settings to initialize the model are missing. Please check the method settings.")
    
    model = ANN(num_inputs, num_hidden_root, num_outputs, global_seed)

    #model = SimpleCNN(num_classes=10)

    return model


def train (model, method_settings, params, HPs, experience, device, global_seed, verbose=0) :
    # Get method settings
    try :
        loss_name = method_settings["loss_name"]
        optimizer_name = method_settings["optimizer_name"]
    except ValueError:
        print("One or more of the required settings to train the model are missing. Please check the method settings.")

    # Get params
    try :
        batch_size = params["batch_size"]
    except ValueError:
        print("One or more of the required params to train the model are missing. Please check the params.")

    # Get HPs
    try :
        #lwf_alpha = HPs["lwf_alpha"]
        #lwf_temperature = HPs["lwf_temperature"]
        num_epochs = HPs["num_epochs"]
    except ValueError:
        print("One or more of the required HPs to train the model are missing. Please check the HPs.")

    # Train    
    lwf = LwF(
        model=model,
        optimizer=get_optimizer(optimizer_name, model),
        criterion=get_loss(loss_name),
        alpha=1, #lwf_alpha,  # Coefficient for distillation loss
        temperature=2, #lwf_temperature,  # Temperature for distillation
        train_mb_size=batch_size,
        train_epochs=num_epochs,
        eval_every=-1,
        device=device)
    

    lwf.train(experience)
    