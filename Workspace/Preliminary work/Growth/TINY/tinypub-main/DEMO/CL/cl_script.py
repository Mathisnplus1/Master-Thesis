import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import copy
import math
from torchinfo import summary

sys.path.append('../../TINY/')
import TINY
import UTILS
import load_data_Loader
import SOLVE_EB as EB
from define_devices import my_device_0, my_device_1


global_starting_time = time.perf_counter()

### activation function with MaxPool

class MaxPoolRelu(torch.nn.Module):
    def __init__(self):
        super(MaxPoolRelu, self).__init__()
        self.kernel_size = 2

    def forward(self, x):
        return torch.nn.ReLU()(torch.nn.MaxPool2d(kernel_size=self.kernel_size)(x))


layer_name = {1: 'CB', 2: 'CB', 3: 'CB', 4: 'CB', 5: 'L', 6: 'L', 7: 'L'}

skeleton = {0: {},
            1: {'in_channels': 3, 'out_channels': 4, 'kernel_size': (3, 3)},
            2: {'in_channels': 4, 'out_channels': 5, 'kernel_size': (3, 3)},
            3: {'in_channels': 5, 'out_channels': 6},
            4: {'in_channels': 6, 'out_channels': 4},
            5: {'size': 10},
            6: {'size': 10},
            7: {'size': 10}}

fct: dict[int, torch.nn.Module] = {depth: torch.nn.ReLU() for depth in range(1, 7)}
fct.update({7: torch.nn.Identity()})
fct.update({2: MaxPoolRelu()})
fct.update({4: MaxPoolRelu()})
to_add = [depth for depth in range(1, 7)]  # depths where neurons are added
to_add_C = [d for d in to_add if
            not (layer_name[d][0] == 'L')]  # depths where neurons are added and are convolutional layer

depth_seuil = {depth: 10 for depth in range(1, 7)}
architecture_growth = 'Our'
rescale = 'DE'
batch_size = 32

dico_parameters = {'skeleton': copy.deepcopy(skeleton),
                   'Loss': UTILS.Loss_entropy,
                   'fct': fct,
                   'layer_name': layer_name,
                   'init_deplacement': 1e-8,
                   'batch_size': batch_size,
                   'lr': 1e-2,
                   'lambda_method': 0.,
                   'accroissement_decay': 1e-8,
                   'lu_conv': 0.001,
                   'max_batch_estimation': 100,
                   'max_amplitude': 20.,
                   'ind_lmbda_shape': 1000,
                   'init_X_shape': [3, 32, 32],
                   'len_train_dataset': 50000,
                   'len_test_dataset': 10000,
                   'T_j_depth': to_add_C,
                   'selection_neuron': UTILS.selection_neuron_seuil,
                   'how_to_define_batchsize': UTILS.indices_non_constant,
                   'depth_seuil': depth_seuil,
                   'architecture_growth': architecture_growth,
                   'rescale': rescale
                   }
CL = TINY.TINY(dico_parameters)

## the Dataset
CL.training_data, CL.test_data = load_data_Loader.load_database_CIFAR10()
CL.tr_loader, CL.te_loader = DataLoader(CL.training_data, batch_size=CL.batch_size, shuffle=True), DataLoader(
    CL.test_data, batch_size=CL.batch_size, shuffle=True)
X, Y = CL.get_batch(data='tr', device=my_device_0)
X_te, Y_te = CL.get_batch(data='te', device=my_device_0)

for k in sorted(list(skeleton.keys()))[1:]:
    if layer_name[k][0] == 'C':
        print('depth ' + str(k) + ' |  Conv | ' + str(skeleton[k]['in_channels']) + ' -> ' + str(
            skeleton[k]['out_channels']))
    else:
        print('depth ' + str(k) + ' | Linear | ' + ' -> ' + str(skeleton[k]['size']))

with torch.no_grad():
    print('Loss on train :', CL.Loss(Y, CL(X)).item(), ' || Loss on test :', CL.Loss(Y_te, CL(X_te)).item())
    print('Accuracy on train :', UTILS.calculate_accuracy(Y, CL(X)), ' || Accuracy on test :',
          UTILS.calculate_accuracy(Y_te, CL(X_te)))

optimizer = torch.optim.SGD(CL.parameters(), lr=CL.lr)
L_tr, L_te, _, A_tr, A_te, _, T = CL.train_batch(optimizer=optimizer, nbr_epoch=0.01)

# df_tracker = pd.DataFrame()
A_tr, A_te, L_tr, L_te, T = np.array([]), np.array([]), np.array([]), np.array([]), np.array([0])

nbr_pass = 2
nbr_epochs_betw_adding = 0.5

path = 'resultats/'


def update_quantity_of_interest():
    global L_tr, L_te, A_tr, A_te, T, df_tracker
    dico_tracker = {'vps' + str(i): [CL.valeurs_propres[i].item()] for i in range(len(CL.valeurs_propres))}

    dico_tracker.update({'depth_add': best_depth, 'nbr_added_neuron': CL.nbr_added_neuron})
    dico_tracker.update(
        {'accroissement': [dico_EB[best_depth]['accroissement']], 'portion_gain': [dico_EB[best_depth]['portion_gain']],
         'nbr_parameters_apres': [CL.count_parameters()], 'nbr_parameters_avant': [nbr_parameters_avant],
         'T': [T[-1]], 'len_L_tr': [len(L_tr)], 'lu_conv': [CL.lu_conv]})

    df_tracker = pd.concat([df_tracker, pd.DataFrame.from_dict(dico_tracker)], ignore_index=True)

    L_tr = np.concatenate([L_tr, l_tr])
    L_te = np.concatenate([L_te, l_te])
    A_tr = np.concatenate([A_tr, a_tr])
    A_te = np.concatenate([A_te, a_te])
    T = np.concatenate([T, t + T[-1]])


def stabilize_training():
    CL.batch_size = math.ceil(np.sqrt(CL.count_parameters() / nbr_parameters_avant) * CL.batch_size)
    # CL.batch_size = math.ceil(CL.count_parameters() / nbr_parameters_avant *  CL.batch_size)


print(summary(CL, input_size=(100, 3, 32, 32)))

count = 1

for j in tqdm(range(nbr_pass)):
    nbr_parameters_avant = CL.count_parameters()
    # gc.collect()
    torch.cuda.empty_cache()

    depth_ajout = to_add  # = [1, 2, 3, 4, 5, 6]
    depth_in_decreasing_criterion, dico_EB = EB.where_is_EB_best_solved(CL, depths=depth_ajout)

    best_depth = depth_in_decreasing_criterion[0]
    dico_EB_bd = dico_EB[best_depth]
    alpha, omega, bias_alpha, vps = dico_EB_bd['alpha'], dico_EB_bd['omega'], dico_EB_bd['bias_alpha'], dico_EB_bd[
        'vps']
    lambda_method = dico_EB_bd['beta_min']

    CL.alpha, CL.omega, CL.bias_alpha, CL.valeurs_propres = None, None, None, []
    CL.TAB_Add = None

    if lambda_method > 0:
        CL.dico_w, CL.lambda_method = dico_EB_bd['dico_w'], dico_EB_bd['beta_min']
        EB.add_neurons(CL, best_depth, alpha=alpha, omega=omega, bias_alpha=bias_alpha, valeurs_propres=vps)
        CL.lambda_method = torch.tensor(0., device=my_device_0)

    stabilize_training()
    CL.tr_loader = DataLoader(CL.training_data, batch_size=CL.batch_size, shuffle=True)
    CL.te_loader = DataLoader(CL.test_data, batch_size=CL.batch_size, shuffle=True)
    optimizer = torch.optim.SGD(CL.parameters(), lr=CL.lr)

    l_tr, l_te, l_va, a_tr, a_te, a_va, t = CL.train_batch(nbr_epochs_betw_adding, optimizer=optimizer)

    # update_quantity_of_interest()

    # df_performance = pd.DataFrame.from_dict({'L_tr' : L_tr, 'L_te' : L_te, 'A_tr' : A_tr, 'A_te' : A_te, 'T' : T[1:],
    #                                          'BatchSize' : np.ones(A_te.shape) * CL.batch_size})
    # df_tracker.to_csv(path + '/df_tracker.csv', index=False)
    # df_performance.to_csv(path + '/df_performance.csv', index=False)
    CL.T = T[-1]
    CL.len_L_tr = len(L_tr)
    # UTILS.save_model_to_file(CL, path = path + '/' , name='model_' + str(count))
    count += 1


print(f'Total time : {time.perf_counter() - global_starting_time:.3e} s')
