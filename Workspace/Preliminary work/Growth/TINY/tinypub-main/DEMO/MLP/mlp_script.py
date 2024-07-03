import sys

sys.path.append('../../TINY/')
import TINY
import UTILS
import math
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import copy
import load_data_Loader
import SOLVE_EB as EB
from define_devices import my_device_0, my_device_1

layer_name = {1: 'L', 2: 'L', 3: 'L'}
skeleton = {0: {}, 1: {'size': 5}, 2: {'size': 5}, 3: {'size': 10}}
fct = {depth: torch.nn.ReLU() for depth in range(1, 3)}
fct.update({3: torch.nn.Identity()})


### la fonction de perte ###
def loss_entropy(x1, x2, reduction='mean'):
    return torch.nn.CrossEntropyLoss(reduction=reduction)(x1, x2)


dico_parameters = {
    'skeleton': copy.deepcopy(skeleton),
    'Loss': loss_entropy,
    'fct': fct,
    'layer_name': layer_name,
    'init_deplacement': 1e-8,  # min amplitude factor
    'batch_size': 64,  # batchsize for training
    'lr': 1e-2,  # leraning rate for training
    'lambda_method': 0,  # = 0 for searching the amplitude factor, if > 0 the
    # amplitude factor is automatically set to this value
    'accroissement_decay': 1e-4,  # the minimum decay to update the
    # architecture with the NewNeurons/BestUpdate
    'depth_seuil': {1: 10, 2: 10},  # maximum number of neurons to add by depth
    'lu_lin': 2,  # reduce the variance of estimators by sqrt of lu_lin > 1
    'max_batch_estimation': 100,  # maximum size of batch at a time
    'max_amplitude': 20.,  # max amplitude factor
    'ind_lmbda_shape': 1000,
    'init_X_shape': [1, 28, 28],  # size of the input, if your X are 1-d, unsqueeze it
    'len_train_dataset': 50000,  # size of training data
    'len_test_dataset': 10000,  # size of testing data
    'selection_neuron': UTILS.selection_neuron_seuil,
    'how_to_define_batchsize': UTILS.indices_non_constant,
}

MLP_model = TINY.TINY(dico_parameters)
MLP_model.training_data, MLP_model.test_data = load_data_Loader.load_database_MNIST(
    batch_size=MLP_model.max_batch_estimation)
MLP_model.tr_loader = DataLoader(MLP_model.training_data, batch_size=MLP_model.batch_size, shuffle=True)
MLP_model.te_loader = DataLoader(MLP_model.test_data, batch_size=MLP_model.batch_size, shuffle=True)

print('MLP_model:\n', MLP_model.layer)

X, Y = MLP_model.get_batch(data='tr', device=my_device_0)  # parametres par défault
X_te, Y_te = MLP_model.get_batch(data='te', device=my_device_0)

optimizer = torch.optim.SGD(MLP_model.parameters(), lr=1e-4)
L_tr, L_te, _, A_tr, A_te, _, T = MLP_model.train_batch(optimizer=optimizer, nbr_epoch=0.01)

df_tracker = pd.DataFrame()
A_tr, A_te, L_tr, L_te, T = np.array([]), np.array([]), np.array([]), np.array([]), np.array([0])
to_add = [1, 2]  # depth where neurons can be added
nbr_pass = 5
nbr_epochs_betw_adding = 0.5  # Time of training between the adding

name_file_expe = 'resultats/'


def update_quantity_of_interest():
    global L_tr, L_te, A_tr, A_te, T, df_tracker
    dico_tracker = {'vps' + str(i): [MLP_model.valeurs_propres[i].item()] for i in
                    range(len(MLP_model.valeurs_propres))}

    dico_tracker.update({'depth_add': best_depth, 'nbr_added_neuron': MLP_model.nbr_added_neuron})
    dico_tracker.update(
        {'accroissement': [dico_EB[best_depth]['accroissement']], 'portion_gain': [dico_EB[best_depth]['portion_gain']],
         'nbr_parameters_apres': [MLP_model.count_parameters()], 'nbr_parameters_avant': [nbr_parameters_avant],
         'T': [T[-1]], 'len_L_tr': [len(L_tr)], 'lu_conv': [MLP_model.lu_conv]})

    df_tracker = pd.concat([df_tracker, pd.DataFrame.from_dict(dico_tracker)], ignore_index=True)

    L_tr = np.concatenate([L_tr, l_tr])
    L_te = np.concatenate([L_te, l_te])
    A_tr = np.concatenate([A_tr, a_tr])
    A_te = np.concatenate([A_te, a_te])
    T = np.concatenate([T, t + T[-1]])


def stabilize_training():
    MLP_model.batch_size = math.ceil(
        np.sqrt(MLP_model.count_parameters() / nbr_parameters_avant) * MLP_model.batch_size)
    # MLP_model.batch_size = math.ceil(MLP_model.count_parameters() / nbr_parameters_avant *  MLP_model.batch_size)


count = 1

for j in tqdm(range(nbr_pass)):
    for k in range(len(to_add)):
        nbr_parameters_avant = MLP_model.count_parameters()
        ## Select the Best depth to add neurons ##
        depth_ajout = to_add  # [1, 2]
        depth_in_decreasing_criterion, dico_EB = EB.where_is_EB_best_solved(MLP_model, depths=depth_ajout)
        best_depth = depth_in_decreasing_criterion[0]
        dico_EB_bd = dico_EB[best_depth]

        alpha = dico_EB_bd['alpha']
        omega = dico_EB_bd['omega']
        bias_alpha = dico_EB_bd['bias_alpha']
        vps = dico_EB_bd['vps']

        lambda_method = dico_EB_bd['beta_min']
        ## Update the model with the NewNeurons
        MLP_model.alpha = None
        MLP_model.omega = None
        MLP_model.bias_alpha = None
        MLP_model.valeurs_propres = []
        MLP_model.TAB_Add = None

        if lambda_method > 0:
            MLP_model.dico_w, MLP_model.lambda_method = dico_EB_bd['dico_w'], dico_EB_bd['beta_min']
            EB.add_neurons(MLP_model, best_depth, alpha=alpha, omega=omega, bias_alpha=bias_alpha, valeurs_propres=vps)
            MLP_model.lambda_method = torch.tensor(0., device=my_device_0)

        ## Training Loops ##
        stabilize_training()
        MLP_model.tr_loader = DataLoader(MLP_model.training_data, batch_size=MLP_model.batch_size, shuffle=True)
        MLP_model.te_loader = DataLoader(MLP_model.test_data, batch_size=MLP_model.batch_size, shuffle=True)
        optimizer = torch.optim.SGD(MLP_model.parameters(), lr=MLP_model.lr)
        l_tr, l_te, l_va, a_tr, a_te, a_va, t = MLP_model.train_batch(nbr_epochs_betw_adding, optimizer=optimizer)

        update_quantity_of_interest()

    # df_performance = pd.DataFrame.from_dict({'L_tr': L_tr, 'L_te': L_te, 'A_tr': A_tr, 'A_te': A_te, 'T': T[1:],
    #                                          'BatchSize': np.ones(A_te.shape) * MLP_model.batch_size})
    # df_tracker.to_csv(name_file_expe + '/df_tracker.csv', index=False)
    # df_performance.to_csv(name_file_expe + '/df_performance.csv', index=False)
    MLP_model.T = T[-1]
    MLP_model.len_L_tr = len(L_tr)
    # UTILS.save_model_to_file(MLP_model, path=name_file_expe + '/', name='model_' + str(count))
    count += 1
