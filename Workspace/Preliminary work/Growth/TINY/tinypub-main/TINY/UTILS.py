#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 13:14:09 2023

@author: verbockhaven
"""
import pickle
import math
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from define_devices import my_device, my_device_0, my_device_1
# from TINY import TINY


###############################################################
###### selection functions ####################################

def selection_neuron_seuil(model: 'TINY', depth: int, TAB=None):
    """
    Select the model.depth_seuil[depth] first neurons (for L-L)
    """
    print('*** ADD : THRESHOLD SELECTION ***')
    # depth_seuil = {2 : 6, 4 : 6, 7 : 12, 9 : 12,
    #              12 : 24, 14 : 24, 17 : 48, 19 : 48}

    seuil = model.depth_seuil[depth]
    model.alpha = model.alpha[:seuil, :]

    if model.layer_name[depth][0] != model.layer_name[depth + 1][0]:  # Conv to Linear
        seuil_CL = seuil * np.array(model.outputs_size_after_activation[depth][:2]).prod()
        model.omega = model.omega[:, :seuil_CL]
    else:
        model.omega = model.omega[:, :seuil]

    model.bias_alpha = model.bias_alpha[: seuil]
    model.valeurs_propres = model.valeurs_propres[: seuil]


def selection_NG_Id(model, depth, TAB=None):
    print('*** NG : NO SELECTION ***')


###############################################################
################## saving the model ###########################

def save_model_to_file(model, path='', name='model'):
    with open(path + name, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)


def read_model_from_file(path='', name='model'):
    file = open(path + name, 'rb')
    return pickle.load(file)


###############################################################
################# Compute the neurons #########################

@torch.no_grad()
def S_1demiN(M: torch.Tensor,
             MDV: torch.Tensor,
             MDV_vrai_gaus: Optional[torch.Tensor] = None,
             architecture_growth: str = 'Our'
             ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Compute U, Sigma and S^(-1/2)N from the matrix S and N

    Parameters
    ----------
    M : torch.Tensor
        The matrix S
    MDV : torch.Tensor
        The matrix N
    MDV_vrai_gaus : torch.Tensor
        I don't know what this is
    architecture_growth : str
        The architecture growth method to use. Can be 'Our' or 'GradMax'

    Returns
    -------
    P : torch.Tensor
        The matrix U of the SVD of S
    D : torch.Tensor
        The vector Sigma of the SVD of S
    S_1demiN : torch.Tensor
        The matrix S^(-1/2)N
    QTV_vrai_gaus : torch.Tensor
        I don't know what this is
    """
    if architecture_growth == 'Our':
        print('*** method : OUR ***')
        try:
            D, P = torch.linalg.eigh(M)
            print('eigh succed')
        except torch.linalg.LinAlgError:
            print('*** FAIL eigh ***')
            D, P = torch.linalg.eigh(M + torch.eye(M.shape[0], device=my_device_0) * max(M.min(), 1e-7))
    elif architecture_growth == 'GradMax':
        print('*** method : GRADMAX ***')
        D, P = torch.ones(M.shape[0], device=my_device_0), torch.eye(M.shape[0], device=my_device_0)
    else:
        raise ValueError(f"Unknown architecture_growth: {architecture_growth}")

    # print('BB^T := PD**2P^T s.t. D[-1]**2 :', D[-1].item(), 'D[0]**2 :', D[0].item())
    index_selected = D > 0
    D[index_selected] = torch.sqrt(D[index_selected])
    index_selected = torch.logical_and(D > 1e-10, index_selected)

    if D[-1] > 0 and (D / D[-1].max() > 1e-7).sum() > 0:
        index_selected = torch.logical_and(D / D[-1].max() > 1e-7, index_selected)
    # print('checks D[idx][-1]**2 :', D[index_selected][-1].item()**2, 'D[idx][0]**2 :', D[index_selected][0].item()**2)

    QTV = torch.matmul(torch.diag(1 / (D[index_selected])),
                       torch.matmul(P[:, index_selected].T, MDV))
    S_1demiN = torch.matmul(P[:, index_selected], QTV)

    if MDV_vrai_gaus is not None:
        QTV_vrai_gaus = torch.matmul(torch.diag(1 / (D[index_selected])),
                                     torch.matmul(P[:, index_selected].T, MDV_vrai_gaus))
        # S_1demiN_gaus = torch.matmul(P[:, index_selected], QTV_vrai_gaus)
    else:
        QTV_vrai_gaus = None

    return P[:, index_selected], D[index_selected], S_1demiN, QTV_vrai_gaus


@torch.no_grad()
def SVD_Smoins1demiN(S_1demiN: torch.Tensor,
                     P: torch.Tensor,
                     D: torch.Tensor
                     ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Compute the optimal neurons weights to add to the network

    Parameters
    ----------
    S_1demiN : torch.Tensor
        The matrix S_1demiN. S^(-1/2)N (where S = (1/n)B_{l-2}B_{l-2}^T and N = (1/n)B_{l-2}(V_goal^proj)^T
    P : torch.Tensor
        The matrix U of the SVD of S
    D : torch.Tensor
        The vector of the singular values of S (i.e. the diagonal of the matrix Sigma of the SVD of S)

    Returns
    -------
    alpha : torch.Tensor
        The optimal neurons weights to add to the network from layer l-2 to layer l-1.
        alpha = sign(lambda) * sqrt(|lambda|) * S^(-1/2) * u
    omega : torch.Tensor
        The optimal neurons weights to add to the network from layer l-1 to layer l.
        omega = sqrt(|lambda|) * v
    eigen_values_lambda : torch.Tensor
        The singular values of S^(-1/2)N noted also lambda
    """
    try:
        u, eigen_values_lambda, omega = torch.linalg.svd(S_1demiN, full_matrices=False)
        print('svd succed')
    except torch.linalg.LinAlgError:
        print('*** FAIL SVD_Smoins1demiN, return 0-vectors ***')
        u = torch.zeros((S_1demiN.shape[0], 1), device=my_device)
        eigen_values_lambda = torch.zeros(1, device=my_device)
        omega = torch.zeros((1, S_1demiN.shape[1]), device=my_device)

    s = torch.sign(eigen_values_lambda)
    sqrt_vps = torch.sqrt(torch.abs(eigen_values_lambda))
    u = u * (s * sqrt_vps).unsqueeze(0)
    omega = sqrt_vps.unsqueeze(1) * omega
    inv_sqrt_of_s = torch.matmul(P, torch.matmul(torch.diag(1 / D), P.T))
    alpha = torch.matmul(inv_sqrt_of_s, u)
    return alpha, omega, eigen_values_lambda


def reshape_neurons(alpha, omega, init_structure, depth, neurontype='LL'):
    if alpha.numel() == 0:
        alpha = torch.tensor([], device=my_device)
        bias_alpha = torch.tensor([], device=my_device)
        omega = torch.tensor([], device=my_device)
    else:
        alias = init_structure[depth]
        alias_2 = init_structure[depth + 1]

        if neurontype == 'CC':
            alpha, bias_alpha = alpha.T[:, :-1], alpha.T[:, -1]
            alpha = alpha.reshape(
                (alpha.shape[0], alias['in_channels'], alias['kernel_size'][0], alias['kernel_size'][1]))
            omega = omega.reshape(
                (omega.shape[0], alias_2['out_channels'], alias_2['kernel_size'][0], alias_2['kernel_size'][1]))
            omega = torch.permute(omega, (1, 0, 2, 3))

        elif neurontype == 'CL':
            alpha, bias_alpha = alpha.T[:, :-1], alpha.T[:, -1]
            alpha = alpha.reshape(
                (alpha.shape[0], alias['in_channels'], alias['kernel_size'][0], alias['kernel_size'][1]))
            # omega = omega.reshape((omega.shape[0], self.outputs_size_after_activation[deep][0] * self.outputs_size_after_activation[deep][1], self.outputs_size_after_activation[deep + 1][0] ))
            omega = omega.reshape(
                (omega.shape[0], int(omega.numel() / (omega.shape[0] * alias_2['size'])), alias_2['size']))
            omega = omega.flatten(start_dim=0, end_dim=1)
            omega = omega.permute((1, 0))

        elif neurontype == 'LL':
            bias_alpha, alpha, omega = alpha.T[:, -1], alpha.T[:, :-1], omega.T
            # bias_alpha, alpha, omega = alpha.T[:, -1], alpha.T[:, :-1], omega.T
        else:
            raise ValueError(f"Unknown neurontype: {neurontype}")

    return alpha, bias_alpha, omega


def condition_valeurs_amplitude_factor(betas, Loss_, accroisement, portion_gain=None):
    if not (portion_gain is None):
        valid_indices = (portion_gain >= 0.)
        betas = betas[valid_indices]
        Loss_ = Loss_[valid_indices]
        portion_gain = portion_gain[valid_indices]

    beta_min = betas[Loss_.argmin()]
    Loss_min = Loss_[Loss_.argmin()]
    ma_portion_gain = portion_gain[Loss_.argmin()]

    mon_accroissement = (Loss_[0] - Loss_min) / torch.abs(Loss_[0])
    print('rate of decrease :', mon_accroissement.item(), 'ampli_fct :', beta_min.item(), 'L[0] :', Loss_[0].item())

    if mon_accroissement < accroisement:
        return torch.tensor(0., device=my_device), mon_accroissement, ma_portion_gain
    else:
        return beta_min, mon_accroissement, ma_portion_gain


###############################################################
#### projection of the desired update #########################

@torch.no_grad()
def layer_w_0_star(w_0_star: Optional[dict[str, torch.Tensor]] = None,
                   padding='same'
                   ) -> None | torch.nn.Conv2d | torch.nn.Linear:
    if w_0_star is not None:
        if len(w_0_star['weight'].shape) == 4:
            in_channels = w_0_star['weight'].shape[1]
            out_channels = w_0_star['weight'].shape[0]
            kernel_size = (w_0_star['weight'].shape[2], w_0_star['weight'].shape[3])

            m = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, device=my_device_1)
            m.weight = torch.nn.parameter.Parameter(w_0_star['weight'].to(my_device_1))
            m.bias = torch.nn.parameter.Parameter(w_0_star['bias'].to(my_device_1))

        elif len(w_0_star['weight'].shape) == 2:
            in_a = w_0_star['weight'].shape[1]
            out_a = w_0_star['weight'].shape[0]

            m = torch.nn.Linear(in_a, out_a, bias=True, device=my_device_1)
            m.weight = torch.nn.parameter.Parameter(w_0_star['weight'].to(my_device_1))
            m.bias = torch.nn.parameter.Parameter(w_0_star['bias'].to(my_device_1))
        else:
            raise ValueError(
                f"w_0_star['weight'] should have 2 or 4 dimensions, but has {len(w_0_star['weight'].shape)}")
        return m
    else:  # WHY can w_0_star be None?
        return None


@torch.no_grad()
def DV_proj(dv: torch.Tensor,
            activities: torch.Tensor,
            m: torch.nn.Module,
            architecture_growth: str = 'Our'):
    # changing the value of DV by projecting DV on the possible parameter space
    # the function change the in-place values of DV, no return
    if m is not None:
        # print('m :', m)
        # print('activities :', activities)
        if architecture_growth == 'Our' or architecture_growth == 'RandomProjection':
            # print('*** DV <- DV_proj ***')
            y = m(activities)
            assert y.shape == dv.shape, \
                f"{y.shape=} != {dv.shape=} this is probably due to incorrect padding"\
                f" in m = {m} which did not correctly transformed activities ({activities.shape=}) to y ({y.shape=})"
            dv.add_(y, alpha=-1)  # DV <- DV - m(activities)


###############################################################
#### eval the performances ####################################


def calculate_accuracy(x1, x2):
    num_classes = min(x1.shape[1], x2.shape[1])
    if x1.shape[1] != x2.shape[1]:
        m = min(x1.shape[1], x2.shape[1])
        x1 = x1[:, : num_classes]
        x2 = x2[:, :num_classes]

    _, inds_pred = torch.max(x1, dim=1)
    _, inds_true = torch.max(x2, dim=1)

    comparaison = (inds_pred == inds_true).int()
    acc = comparaison.sum() / len(comparaison)
    acc_cpu = acc.cpu().item()
    del acc
    del inds_pred, inds_true
    torch.cuda.empty_cache()
    return acc_cpu


def topk_accuracy(x1, x2, k=5):
    num_classes = min(x1.shape[1], x2.shape[1])
    if x1.shape[1] != x2.shape[1]:
        m = min(x1.shape[1], x2.shape[1])
        x1 = x1[:, : num_classes]
        x2 = x2[:, :num_classes]

    _, inds_pred = torch.sort(x1, dim=1, descending=True)
    _, inds_true = torch.max(x2, dim=1)

    s = 0
    for i in range(k):
        s += (inds_pred[:, i] == inds_true).sum() * 1.
    acc = s / x1.shape[0]
    acc_cpu = acc.cpu().item()
    del acc
    del inds_pred, inds_true
    torch.cuda.empty_cache()
    return acc_cpu


def Loss_entropy(x1, x2, reduction='mean'):
    return torch.nn.CrossEntropyLoss(reduction=reduction)(x1, x2)


###############################################################
#### Estimate the size of the batchsize for the estimations ###

coef_lu_NG = {32 * 32: 1., 16 * 16: 1.6841667, 8 * 8: 1.6841667 * 2.4240465,
              4 * 4: 1.6841667 * 2.4240465 * 2.3486185}
coef_lu_Add = {1024: 1., 256: 2.0667207, 64: 2.0667207 * 2.0450544,
               16: 2.0667207 * 2.0450544 * 2.6325793}

# valeur au pif
coef_lu_NG[900] = 5
coef_lu_Add[900] = 5

coef_lu_NG[784] = 5
coef_lu_Add[784] = 5


def indices_non_constant(model, depth, method=None):
    """
    Compute the size of the batchsize for the estimations

    Depend of the number of neurons in the layers and the method used
    """
    size_batch = 1.
    if method == 'NG':
        if model.layer_name[depth][0] == 'C':
            # print('cles coelf NG :', model.outputs_size_after_activation[depth - 1][0] * model.outputs_size_after_activation[depth - 1][1])
            size_batch *= coef_lu_NG[model.outputs_size_after_activation[depth - 1][0] *
                                     model.outputs_size_after_activation[depth - 1][1]] ** 2
            size_batch *= model.layer[depth]['C'].in_channels * model.layer[depth]['C'].out_channels
        else:
            size_batch *= model.layer[depth]['L'].in_features * model.layer[depth]['L'].out_features
    elif method == 'Add':
        if model.layer_name[depth + 1][0] == 'C':
            size_batch *= coef_lu_Add[model.outputs_size_after_activation[depth][0] *
                                      model.outputs_size_after_activation[depth][1]] ** 2
            size_batch *= model.layer[depth]['C'].in_channels * model.layer[depth + 1]['C'].out_channels
        elif model.layer_name[depth][0] == 'L':
            size_batch *= model.layer[depth]['L'].in_features * model.layer[depth + 1]['L'].out_features
        else:
            co = np.array(model.outputs_size_after_activation[depth][:2]).prod()
            print('co :', co)
            size_batch *= co * model.layer[depth]['C'].in_channels * model.layer[depth + 1]['L'].out_features
    # print('in_channels x out_channels x coef_lu :', size_batch)
    if method == 'Add' and model.layer_name[depth][0] == 'C' and model.layer_name[depth + 1][0] == 'L':
        size_batch *= model.lu_conv_lin
    elif model.layer_name[depth][0] == 'C':
        size_batch *= model.lu_conv
        # print('x :', model.lu_conv)
    elif model.layer_name[depth][0] == 'L':
        size_batch *= model.lu_lin
        # print('x :', model.lu_lin)

    size_batch = int(size_batch)

    if not (getattr(model, 'tr_loader', None) is None):
        model.tr_loader = torch.utils.data.DataLoader(model.training_data, batch_size=model.max_batch_estimation,
                                                         shuffle=True)
        model.te_loader = torch.utils.data.DataLoader(model.test_data, batch_size=model.max_batch_estimation,
                                                         shuffle=True)

    size_batch = max(200, size_batch)
    l = math.ceil(size_batch / model.max_batch_estimation) * model.max_batch_estimation

    if not (getattr(model, 'tr_loader', None) is None):
        max_len_l = max(model.len_train_dataset, l)
    else:
        max_len_l = l

    if not (getattr(model, 'tr_loader', None) is None):
        max_len_l = min(model.len_train_dataset, l)
    else:
        max_len_l = max(model.len_train_dataset, l)

    model.ind = torch.randperm(max_len_l)[:l]
    model.seed = torch.randperm(max_len_l)[:l]
    model.ind_lmbda = torch.randperm(max_len_l)[:model.ind_lmbda_shape]
    model.seed_lmbda = torch.randperm(max_len_l)[:model.ind_lmbda_shape]

    # print('batch size for estimation :', model.ind.shape[0])


###############################################################
#### plotting functions  ######################################

def lisser_courbe(X, Y=None, coef=5):
    X_lisse = []
    for j in range(int(X.shape[0] / coef)):
        X_lisse.append(X[j * coef: (j + 1) * coef].mean())
    return np.array(X_lisse)


def zoom_courbe(X, Y=None, start=0):
    assert start >= 0 and start <= 1

    index_start = int(X.shape[0] * start)
    return X[index_start:]


###############################################################
############### Construction of the model #####################


def construct_output_size(model):
    init_X_shape = model.init_X_shape
    model.outputs_size_after_activation = {}

    if len(init_X_shape) == 3:
        model.outputs_size_after_activation[0] = [init_X_shape[1], init_X_shape[2], init_X_shape[0]]
    else:
        model.outputs_size_after_activation[0] = [init_X_shape[0]]

    def register_size(module, i, o):
        # dico_size[model.where] = o.shape
        if len(o.shape) == 4:
            model.outputs_size_after_activation[model.where] = [o.shape[2], o.shape[3], o.shape[1]]
        else:
            model.outputs_size_after_activation[model.where] = [o.shape[1]]

    size = (2,) + init_X_shape
    X = torch.zeros(size, device=my_device_0)
    model.rf = {}

    for d in range(1, model.deep + 1):
        # print('d :', d)
        model.rf[d] = model.fct[d].register_forward_hook(register_size)
    with torch.no_grad():
        model(X)

    for d in range(1, model.deep + 1):
        model.rf[d].remove()
    del model.rf
    # print('outputs_size_after_activation :', model.outputs_size_after_activation)
