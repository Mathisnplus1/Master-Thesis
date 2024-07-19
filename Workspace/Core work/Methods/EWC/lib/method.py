from lib.models import ANN
from abstract_torch import get_optimizer, get_loss
import numpy as np

from avalanche.training.supervised import EWC


def initialize_model(method_settings, global_seed) :
    try :
        num_inputs = method_settings["num_inputs"]
        num_hidden_root = method_settings["num_hidden_root"]
        num_outputs = method_settings["num_outputs"]
    except ValueError:
        print("One or more of the required settings to initialize the model are missing. Please check the method settings.")
    
    model = ANN(num_inputs, num_hidden_root, num_outputs, global_seed)

    return model


def initialize_training(model, method_settings, benchmark_settings, device) :
    try :
        optimizer_name = method_settings["optimizer_name"]
        loss_name = method_settings["loss_name"]
    except ValueError:
        print("One or more of the required settings to initialize the HPO are missing. Please check the method settings.")
    try :
        batch_size = benchmark_settings["batch_size"]
    except ValueError:
        print("One or more of the required settings to initialize the HPO are missing. Please check the benchmark settings.")

    ewc = EWC(
        model=model, 
        optimizer=get_optimizer(optimizer_name, model),
        criterion=get_loss(loss_name),
        ewc_lambda=0,
        train_mb_size=batch_size,
        train_epochs=0, 
        eval_every=-1,
        device=device
    )

    return ewc


def train (model, method_settings, params, HPs, experience, device, global_seed, verbose=0) :
    # Get HPs
    try :
        ewc_lambda = HPs["ewc_lambda"]
        num_epochs = HPs["num_epochs"]
    except ValueError:
        print("One or more of the required HPs to train the model are missing. Please check the HPs.")
    # Get params
    try :
        ewc = params["ewc"]
    except ValueError:
        print("One or more of the required params to train the model are missing. Please check the params.")

    # Train
    ewc.model = model
    ewc.optimizer = get_optimizer(method_settings["optimizer_name"], model)
    ewc.plugins[0].ewc_lambda = ewc_lambda
    ewc.train_epochs = num_epochs

    ewc.train(experience)

    return ewc