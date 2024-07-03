#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:10:59 2023

@author: verbockhaven
"""
from typing import Optional, Iterable
import time
import copy
import gc
import math
import numpy as np
import torch
from define_devices import my_device_0, my_device_1, my_device

import GLOBALS
import T_S_F_N
import UTILS


class my_linear(torch.nn.Module):
    def __init__(self):
        super(my_linear, self).__init__()

    def forward(self, x, w, b):
        return torch.nn.functional.linear(x, weight=w, bias=b)


class my_conv2d(torch.nn.Module):
    def __init__(self):
        super(my_conv2d, self).__init__()

    def forward(self, x, w, b, padding='same', stride=1):
        return torch.nn.functional.conv2d(x, weight=w, bias=b,
                                          padding=padding, stride=stride)


# TODO : fix the following
X_train_rescale = None
Y_train_rescale = None
X_test_rescale = None
Y_test_rescale = None


class TINY(torch.nn.Module):

    def __init__(self,
                 parameters: dict,
                 batch_size: int = 128,
                 lr: float = 1e-2,
                 gradient_clip: Optional[float] = None,
                 scheduler: Optional[callable] = None,
                 len_train_dataset: int = 50_000,
                 len_test_dataset: int = 10_000,
                 loss: torch.nn.modules.loss = torch.nn.MSELoss(reduction='sum'),
                 skeleton: dict = None,
                 layer_name: dict = None,
                 activation_function: dict[str, torch.nn.Module] = None,
                 skip_connections: dict[str, tuple[int, int]] = None,
                 init_input_x_shape: tuple[int, int, int] = (3, 32, 32),
                 skip_functions: dict[int, torch.nn.Module] = None,
                 T_j_depth: Optional[list[int]] = None,
                 lambda_method: float = 0.,
                 lambda_method_NG: float = 0.,
                 init_deplacement: float = 1e-8,
                 init_deplacement_NG: Optional[float] = None,
                 accroissement_decay: float = 1e-3,
                 accroissement_decay_NG: Optional[float] = None,
                 exp: int = 2,
                 ind_lmbda_shape: int = 1_000,
                 max_amplitude: float = 1.,
                 rescale: str = 'DE',
                 architecture_growth: str = 'Our',
                 selection_neuron: callable = UTILS.selection_neuron_seuil,
                 selection_NG: callable = UTILS.selection_NG_Id,
                 ):

        super(TINY, self).__init__()
        local_time = time.time()

        # training parameters ##
        self.batch_size = parameters.get('batch_size', batch_size)
        self.lr = parameters.get('lr', lr)
        self.gradient_clip = parameters.get('gradient_clip', gradient_clip)
        self.scheduler = parameters.get('scheduler', scheduler)
        self.len_train_dataset = parameters.get('len_train_dataset', len_train_dataset)
        self.len_test_dataset = parameters.get('len_test_dataset', len_test_dataset)
        self.Loss = parameters.get('Loss', loss)
        self.key_order = {'CBR': ['C', 'B', 'R'], 'LD': ['D', 'L'], 'L': ['L'], 'CB': ['C', 'B']}

        # structure ##
        assert skeleton is not None or 'skeleton' in parameters, 'skeleton is missing'
        self.skeleton = copy.deepcopy(parameters['skeleton']) if skeleton is None else copy.deepcopy(skeleton)
        self.deep = len(self.skeleton) - 1
        assert layer_name is not None or 'layer_name' in parameters, 'layer_name is missing'
        self.layer_name = parameters.get('layer_name', layer_name)
        self.break_conv = list(self.layer_name.values()).count('CBR') + \
                          list(self.layer_name.values()).count('C') + \
                          list(self.layer_name.values()).count('CB') + 1
        self.layer = {}
        self.fct = parameters.get('fct', {depth: torch.nn.ReLU() for depth in range(1, self.deep + 1)}
                                  if activation_function is None else activation_function)
        self.skip_connections = parameters.get('skip_connections', {'in': [], 'out': []} if skip_connections is None else skip_connections)
        self.init_X_shape = tuple(parameters.get('init_X_shape', init_input_x_shape))
        self.skip_fct = {depth: parameters.get('skip_fct', {} if skip_functions is None else skip_functions).get(depth, torch.nn.Identity())
                         for depth in self.skip_connections['in']}
        self.T_j_depth = parameters.get('T_j_depth', list(range(1, self.break_conv - 1)) if T_j_depth is None else T_j_depth)

        # amplitude factor ##
        # fixed #
        self.lambda_method = torch.tensor(parameters.get('lambda_method', lambda_method), device=my_device_0)
        self.lambda_method_NG = torch.tensor(parameters.get('lambda_method_NG', lambda_method_NG), device=my_device_0)
        self.init_deplacement = torch.tensor(parameters.get('init_deplacement', init_deplacement), device=my_device_0)
        self.init_deplacement_NG = torch.tensor(
            parameters.get('init_deplacement_NG',
                           init_deplacement if init_deplacement_NG is None else init_deplacement_NG), device=my_device_0)
        self.accroissement_decay = parameters.get('accroissement_decay', accroissement_decay)
        self.accroissement_decay_NG = parameters.get('accroissement_decay_NG',
                                                     accroissement_decay if accroissement_decay_NG is None
                                                     else accroissement_decay_NG)
        self.exp = parameters.get('exp', exp)
        self.ind_lmbda_shape = parameters.get('ind_lmbda_shape', ind_lmbda_shape)
        # TODO: check if the two following parameters are needed
        self.ind_lmbda = parameters.get('ind_lmbda', torch.arange(self.ind_lmbda_shape))
        self.seed_lmbda = parameters.get('ind_lmbda', torch.arange(self.ind_lmbda_shape))
        self.max_amplitude = parameters.get('max_amplitude', max_amplitude)
        self.rescale = parameters.get('rescale', rescale)

        # to compute #
        self.amplitude_factor = torch.tensor(0., device=my_device_0)
        self.accroissement = torch.tensor(0., device=my_device_0)
        self.accroissement_NG = torch.tensor(0., device=my_device_0)
        self.beta_min = torch.tensor(0.)

        # adding neurons parameters ##
        # fixed #
        self.architecture_growth = parameters.get('architecture_growth', architecture_growth)
        self.selection_neuron = parameters.get('selection_neuron', selection_neuron)
        self.selection_NG = parameters.get('selection_NG', selection_NG)
        # TODO: continue replacing the following parameters with the ones in the parameters dict
        self.ind = parameters.get('ind', torch.arange(1000))
        self.seed = parameters.get('ind', torch.arange(1000))
        self.f_conv = my_conv2d()
        self.f_lin = my_linear()
        self.h = {}
        self.depth_seuil = parameters.get('depth_seuil', {depth: 10 for depth in self.layer_name.keys()})

        # to compute #
        self.dico_w = None
        self.alpha_computed = None
        self.omega_computed = None
        self.valeurs_propres_computed = None
        self.alpha = None
        self.bias_alpha = None
        self.omega = None
        self.valeurs_propres = None
        self.nbr_added_neuron = 0

        # statistics ##
        self.random_M = parameters.get('random_M', False)
        self.random_activity = parameters.get('random_activity', False)
        # self.do_F_test = parameters.get('do_F_test', False)
        self.how_to_define_batchsize = parameters.get('how_to_define_batchsize', None)
        self.lu_conv = parameters.get('lu_conv', 0.5)
        self.lu_lin = parameters.get('lu_lin', 2)
        self.lu_conv_lin = parameters.get('lu_conv_lin', 1)
        self.where = None

        # manage memory ##
        self.max_batch_estimation = parameters.get('max_batch_estimation', 10_000)

        # construction of the layers ####
        x = torch.randn((1,) + self.init_X_shape, device=my_device_0)
        self.skeleton[0]['size'] = x[0].numel()
        for j in range(1, self.deep + 1):
            if self.layer_name[j] == 'LD':
                self.layer[j] = {'D': torch.nn.Dropout(0.5),
                                 'L': torch.nn.Linear(self.skeleton[j - 1]['size'], self.skeleton[j]['size'], bias=True,
                                                      device=my_device_0)}
            elif self.layer_name[j] == 'L':
                self.layer[j] = {'L': torch.nn.Linear(self.skeleton[j - 1]['size'], self.skeleton[j]['size'], bias=True,
                                                      device=my_device_0)}
            elif self.layer_name[j] == 'CB':
                self.skeleton[j]['kernel_size'] = self.skeleton[j].get('kernel_size', (3, 3))
                self.layer[j] = {'C': torch.nn.Conv2d(self.skeleton[j]['in_channels'], self.skeleton[j]['out_channels'],
                                                      self.skeleton[j]['kernel_size'],
                                                      padding=self.skeleton[j].get('padding', 1),
                                                      stride=self.skeleton[j].get('stride', 1),
                                                      bias=self.skeleton[j].get('bias', True), device=my_device_0),
                                 # 'B' :  torch.nn.Identity(),
                                 'B': torch.nn.BatchNorm2d(self.skeleton[j]['out_channels'], momentum=0.1,
                                                           device=my_device_0)
                                 }

            with torch.no_grad():
                if j <= self.deep and self.layer_name[j][0] == 'L':
                    x = x.flatten(start_dim=1)

                for k in self.layer[j].keys():
                    x = self.layer[j][k](x)
                x = self.fct[j](x)

                print(f"After layer {j} : {x.shape=}")

            self.skeleton[j]['size'] = x[0].numel()

        UTILS.construct_output_size(self)

        # construction of the T,F and S matrices ####

        for j in self.T_j_depth:
            GLOBALS.dico_mask_tensor_t[j] = T_S_F_N.compute_mask_tensor_t(self, j)
            GLOBALS.dico_tt[j] = T_S_F_N.compute_tensor_tt(None, None, GLOBALS.dico_mask_tensor_t[j])
            GLOBALS.dico_F[j] = T_S_F_N.creation_T_C_pour_BCR(self, j)
            gc.collect()
            torch.cuda.empty_cache()
        if self.break_conv > 1:
            GLOBALS.dico_F[self.break_conv - 1] = T_S_F_N.creation_T_C_pour_BCR(self, self.break_conv - 1)

        for d in GLOBALS.dico_mask_tensor_t.keys():
            GLOBALS.dico_T_permute_flat[d] = GLOBALS.dico_mask_tensor_t[d].transpose(1, 2).to_dense().flatten(start_dim=0,
                                                                                                              end_dim=1).to_sparse()
            GLOBALS.dico_T_flat_sparse[d] = GLOBALS.dico_mask_tensor_t[d].to_dense().flatten(start_dim=0, end_dim=1).to_sparse()

        print(f"Time to build the model: {time.time() - local_time:.2f} seconds")

    def parameters(self, recurse=True) -> Iterable[torch.nn.Parameter]:
        """
        Gets the parameters of self.
        
        Returns :
        list of tensor
        """
        # params = []
        for layer in self.layer.values():
            for k in layer.keys():
                for p in layer[k].parameters():
                    # params.append(p)
                    yield p
        if not (getattr(self, 'skip_fct', None) is None):
            for RS_l in self.skip_fct.values():
                for p in RS_l.parameters():
                    # params.append(p)
                    yield p
        # return params

    def disable_gradient(self):
        """
        Disables de gradient of all parameters of self.
        """
        for p in self.parameters():
            p.requires_grad = False

    def able_gradient(self):
        """
        Put a gradient flag on all parameters of self.
        """
        for p in self.parameters():
            p.requires_grad = True

    def get_batch(self, data='tr', indices=None, device=my_device_0, seed: Optional[int] = None
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a batch of inputs and outputs from the train/test distribution.
        """
        if getattr(self, data + '_loader', None) is None:
            if data == 'tr':
                DX, DY = X_train_rescale, Y_train_rescale
            else:
                DX, DY = X_test_rescale, Y_test_rescale
            input_x = DX[indices].to(device)
            target_y = DY[indices].to(device)
        else:
            if data == 'tr':
                loader = self.tr_loader
            else:
                loader = self.te_loader
            if not (seed is None):
                torch.manual_seed(seed[0])
            input_x, target_y = next(iter(loader))
            input_x = input_x.to(device)
            target_y = target_y.to(device)
        return input_x, target_y

    def count_parameters_layer(self, depth: int) -> int:
        """
        Counts the parameters of self at a specific depth.
        
        Returns :
        int
        """
        return sum(p.numel() for key in self.layer[depth].keys() for p in self.layer[depth][key].parameters())

    def count_parameters(self) -> int:
        """
        Counts the parameters of self.
        """
        return sum(p.numel() for p in self.parameters())

    def initialize(self) -> None:
        """
        Initialize the parameters of self.
        """
        for key in self.layer.keys():
            for sous_key in self.layer[key].keys():
                if isinstance(self.layer[key][sous_key], torch.nn.Conv2d) or isinstance(self.layer[key][sous_key],
                                                                                        torch.nn.Linear):
                    torch.nn.init.xavier_normal_(self.layer[key][sous_key].weight,
                                                 gain=torch.nn.init.calculate_gain('relu'))
                    torch.nn.init.zeros_(self.layer[key][sous_key].bias)

    def complexity_model(self):
        """
        Computes the time complexity in term of additions of self at test.
        """
        complexity = 0
        # Im_size_c = 3
        Im_size_2 = 32 * 32
        for depth in range(1, self.deep + 1):
            if self.layer_name[depth][0] == 'C':
                Im_size_c = self.layer[depth]['C'].in_channels
                out_c = self.layer[depth]['C'].out_channels
                complexity += 9 * Im_size_c * Im_size_2 * out_c

                if depth in list(GLOBALS.dico_F.keys()):
                    Im_size_2 /= 4
            else:
                complexity += self.layer[depth]['L'].weight.numel()
        return complexity

    ############# forward ###############################
    #####################################################

    def forward(self, x, depth_break: Optional[int] = None) -> torch.Tensor:
        self.where = 1
        stop = False
        while self.where <= self.deep and self.layer_name[self.where][0] == 'C' and not stop:
            if self.skip_connections is not None and self.where in self.skip_connections['in']:
                clone_x = self.skip_fct[self.where](x.clone())
                # print('where, clone_x.shape :', self.where, clone_x.shape)
            else:
                clone_x = 0

            for key in self.key_order[self.layer_name[self.where]]:

                if getattr(self.layer[self.where][key], 'inplace', False):
                    self.layer[self.where][key](x)
                else:
                    x = self.layer[self.where][key](x)
            if self.skip_connections is not None and self.where in self.skip_connections['out']:
                x = x + clone_x

            if getattr(self.fct[self.where], 'inplace', False):
                self.fct[self.where](x)
            else:
                x = self.fct[self.where](x)

            if depth_break is not None and depth_break == self.where:
                stop = True
            # print(x.shape)
            self.where += 1

        if not stop:
            x = torch.flatten(x, start_dim=1)

        while self.where <= self.deep and not stop:
            for key in self.key_order[self.layer_name[self.where]]:
                if getattr(self.layer[self.where][key], 'inplace', False):
                    self.layer[self.where][key](x)
                else:
                    x = self.layer[self.where][key](x)

            if getattr(self.fct[self.where], 'inplace', False):
                self.fct[self.where](x)
            else:
                x = self.fct[self.where](x)

            self.where += 1
        return x

    @torch.no_grad()
    def forward_naturel(self, x, depth=1, lmbda=1e-3):
        """
        Computes the output of the network considering the update of its parameters at depth with the natural gradient,
        and the norm of the functional update induces by this update of parameters.
        """
        self.NGhook = self.layer[depth][self.layer_name[depth][0]].register_forward_hook(self.__BestUpdateHook)
        self.ampl_fact_temp = lmbda
        y = self(x)
        norm = self.norm_x_over_norm_DE
        self.NGhook.remove()
        return y, norm

    @torch.no_grad()
    def forward_ajout(self, x, lmbda=torch.tensor(1., device=my_device_0), depth=1):
        """
        Computes the output of the network with the NewNeurons at depth,
        and the norm of the functional update induces by this add.
        """
        self.hAddhookr = self.layer[depth][self.layer_name[depth][0]].register_forward_hook(self.AddHookleft)
        self.hAddhookl = self.layer[depth + 1][self.layer_name[depth + 1][0]].register_forward_hook(self.AddHookright)
        self.copy_fct = copy.deepcopy(self.fct[depth])
        self.hAddHookfct = self.fct[depth].register_forward_hook(self.AddHookfct)
        if 'B' in self.layer_name[depth]:
            self.hAddHookBN = self.layer[depth]['B'].register_forward_hook(self.AddHookBN)

        self.ampl_fact_temp = lmbda.item()
        with torch.no_grad():
            y_add = self(x)

        self.hAddhookr.remove()
        self.hAddhookl.remove()
        self.hAddHookfct.remove()
        if 'B' in self.layer_name[depth]:
            self.hAddHookBN.remove()

        return y_add, self.norm_x_over_norm_DE_left, self.norm_x_over_norm_DE_right

    #######################################################
    ##################### desired update ##################

    def __deplacement_voulu_random(self, depth: int, ind_MB=None, X=None, Y=None):
        if X is None:
            X = X_train_rescale[ind_MB].to(my_device)
        self(X, depth_break=depth + 1)

        if self.layer_name[depth][0] == 'C':
            # assert self.layer[DV_deep][self.layer_name[DV_deep][0]].padding == 'same'

            channels = [self.layer[depth][self.layer_name[depth][0]].out_channels]
            IM_shape = self.outputs_size_after_activation[depth - 1][:2]
            size = [X.shape[0]] + channels + IM_shape
        else:
            size = [X.shape[0]] + [self.layer[depth][self.layer_name[depth][0]].out_features]

        return torch.randn(size, device=my_device_1)

    def __deplacement_voulu_manifold(self,
                                     depth: int,
                                     input_x: Optional[torch.Tensor] = None,
                                     target_y: Optional[torch.Tensor] = None,
                                     ind_MB = None,
                                     ) -> torch.Tensor:
        if input_x is None:
            input_x, target_y = self.get_batch(indices=ind_MB, device=my_device_1)

        self.disable_gradient()
        reg_out = self.layer[depth][self.layer_name[depth][0]].register_forward_hook(
            self.__activities_output_with_gradient)
        loss = self.Loss(self(input_x), target_y)
        reg_out.remove()
        if self.Loss(torch.ones((1, 1)) * 1.0, torch.zeros((1, 1))) ==\
           self.Loss(torch.ones((2, 1)) * 1.0, torch.zeros((2, 1))):  # test if the loss is sum or mean
            _DV = torch.autograd.grad(loss, GLOBALS.outputs[depth], retain_graph=False)[0] * target_y.shape[0]

        else:
            _DV = torch.autograd.grad(loss, GLOBALS.outputs[depth])[0]
        self.able_gradient()

        return _DV

    def deplacement_voulu(self, DV_deep, ind_MB=None, X=None, Y=None):
        """
        Computes the desired update at depth. (Vgoal)
        """
        if self.architecture_growth == 'RandomProjection':
            # print('DV random :(')
            return self.__deplacement_voulu_random(DV_deep, ind_MB=ind_MB, X=X, Y=Y)
        elif self.architecture_growth == 'Our' or self.architecture_growth == 'GradMax':
            return self.__deplacement_voulu_manifold(DV_deep, input_x=X, target_y=Y, ind_MB=ind_MB)
        else:
            raise ValueError(f"architecture_growth: {self.architecture_growth} not recognized")

    ###############################################################
    ########## changing the values of the BestUpdate/NewNeurons ###
    @torch.no_grad()
    def rescale_dico_w(self, depth_NG):
        """
        Normalizes the BestUpdate.
        """
        if self.rescale == 'DE':
            print('*** normalize BestUpdate acc. ||functional udpate|| ***')
            x, y_true = self.get_batch(device=my_device_0)
            with torch.no_grad():
                y, norm_x_over_norm_DE = self.forward_naturel(x, depth_NG, lmbda=1e-1)

            self.dico_w['weight'] *= norm_x_over_norm_DE
            self.dico_w['bias'] *= norm_x_over_norm_DE
        else:
            print('*** normalize BestUpdate acc. ||parameters|| ***')
            norm_new_weight = torch.cat([self.dico_w['weight'].flatten(), self.dico_w['bias']]).norm()
            coeff = torch.sqrt(
                self.layer[depth_NG][self.layer_name[depth_NG][0]].weight.norm() ** 2 + self.layer[depth_NG][
                    self.layer_name[depth_NG][0]].bias.norm() ** 2) / torch.sqrt(norm_new_weight)
            self.dico_w['weight'], self.dico_w['bias'] = self.dico_w['weight'] * coeff, self.dico_w['bias'] * coeff

    @torch.no_grad()
    def rescale_alpha_omega(self, depth_ajout):
        """
        Normalizes the parameters of the NewNeurons.
        """
        if self.rescale == 'DE':
            print('*** normalize NewNeurons acc. ||functional udpate|| ***')
            x, y_true = self.get_batch(indices=torch.arange(100), device=my_device_0)
            with torch.no_grad():
                lmbda = torch.tensor(1., device=my_device_0)
                y, norm_x_over_norm_DE_depth, norm_x_over_norm_DE_depthp1 = self.forward_ajout(x, depth=depth_ajout,
                                                                                               lmbda=lmbda)
            print('norm_x_over_norm_DE_depth:', norm_x_over_norm_DE_depth.item())
            print('norm_x_over_norm_DE_depthp1 : ', norm_x_over_norm_DE_depthp1.item())
            self.alpha *= norm_x_over_norm_DE_depth
            self.bias_alpha *= norm_x_over_norm_DE_depth
            self.omega *= norm_x_over_norm_DE_depthp1
        else:
            print('***  normalize NewNeurons acc. ||parameters|| ***')
            # norm_layer_l_1 = (self.layer[depth_ajout][self.layer_name[depth_ajout][0]].weight.norm() ** 2) / self.layer[depth_ajout][self.layer_name[depth_ajout][0]].weight.shape[0]
            # norm_layer_l = (self.layer[depth_ajout][self.layer_name[depth_ajout][0]].weight.norm() ** 2) / self.layer[depth_ajout][self.layer_name[depth_ajout][0]].weight.shape[1]
            norm_layer_l_1, norm_layer_l = 1., 1.
            norm_alpha = (self.alpha.norm() ** 2) / self.alpha.shape[0]
            norm_omega = (self.omega.norm() ** 2) / self.omega.shape[1]

            coeff_alpha = torch.sqrt(norm_layer_l_1 / norm_alpha)
            coeff_omega = torch.sqrt(norm_layer_l / norm_omega)
            self.alpha, self.bias_alpha = self.alpha * coeff_alpha, self.bias_alpha * coeff_alpha
            self.omega = self.omega * coeff_omega

    @torch.no_grad()
    def alpha_omega_sign(self, depth):
        """
        Choose the sign of the parameters of the NewNeurons.
        """
        x, y_true = self.get_batch(indices=torch.arange(100), device=my_device_0)

        with torch.no_grad():
            L_plus = self.Loss(self.forward_ajout(x, lmbda=self.init_deplacement, depth=depth)[0], y_true)
            self.alpha.mul_(-1), self.bias_alpha.mul_(-1), self.omega.mul_(-1)
            L_moins = self.Loss(self.forward_ajout(x, lmbda=self.init_deplacement, depth=depth)[0], y_true)

            if L_plus < L_moins:
                print('L_plus < L_moins:', L_plus.item(), '<', L_moins.item())
                self.alpha.mul_(-1), self.bias_alpha.mul_(-1), self.omega.mul_(-1)
            else:
                print('(alpha, omega) <-- (-alpha, -omega)')

    @torch.no_grad()
    def scale_new_neurons(self, ampl):
        """
        Multiplies the parameters of the NewNeurons with the amplitude factor.
        """
        print('amplitude factor for the new neurons :', ampl.item())
        if self.architecture_growth == 'Our':
            print('(alpha, omega) <-- (sqrt(ampl) x alpha, sqrt(ampl) x omega)')
            self.alpha.mul_(torch.sqrt(ampl))
            self.bias_alpha.mul_(torch.sqrt(ampl))
            self.omega.mul_(torch.sqrt(ampl))

        elif self.architecture_growth == 'GradMax':
            print('(alpha, omega) <-- (0, ampl x omega)')
            self.alpha *= 0.
            self.bias_alpha *= 0.
            self.omega *= ampl

        elif self.architecture_growth == 'RandomProjection':
            print('(alpha, omega) <-- (ampl x alpha, 0)')
            self.omega *= 0
            self.alpha *= ampl
            self.bias_alpha *= ampl
        else:
            raise ValueError(f"architecture_growth: {self.architecture_growth} not recognized")

    ##############################################
    ######### update the architecture ############

    def gradient_naturel(self, depth, update, lmbda=1.):
        """
        Change the parameters of self at depth with the dictionnary update.
        """
        with torch.no_grad():
            lettre = self.layer_name[depth][0]
            save_weight, save_bias = self.layer[depth][lettre].weight.detach(), self.layer[depth][lettre].bias.detach()
            del self.layer[depth][lettre].weight, self.layer[depth][lettre].bias

            self.layer[depth][lettre].weight = torch.nn.parameter.Parameter(lmbda * update['weight'] + save_weight,
                                                                            requires_grad=True)
            self.layer[depth][lettre].bias = torch.nn.parameter.Parameter(lmbda * update['bias'] + save_bias,
                                                                          requires_grad=True)

            setattr(self.layer[depth][lettre].weight, 'requires_grad', True)
            setattr(self.layer[depth][lettre].bias, 'requires_grad', True)

            if not (getattr(self, 'optimizer', None) is None):
                self.optimizer.gradient_naturel(self, depth, amplitude_factor=lmbda)

    def update_architecture(self, deep, alpha_shape, omega_shape):
        """
        Change the attributs skeleton and outputs_size_after_activation 
        of self according to number of neurons added.
        """
        if self.layer_name[deep][0] == 'C':

            self.skeleton[deep]['size'] = int(self.skeleton[deep]['size'] / self.skeleton[deep]['out_channels'] * (
                        alpha_shape[0] + self.skeleton[deep]['out_channels']))
            self.skeleton[deep]['out_channels'] += alpha_shape[0]
            self.outputs_size_after_activation[deep][2] += alpha_shape[0]
        else:
            self.skeleton[deep]['size'] += alpha_shape[0]
            self.outputs_size_after_activation[deep][0] += alpha_shape[0]

        if self.layer_name[deep + 1][0] == 'C':
            self.skeleton[deep + 1]['in_channels'] += alpha_shape[0]

    def help_add_K_neurons_linear_right(self, depht, weight_to_add, lambda_w=1):
        """
        Add the outgoing weights of the added neurons neurons for a linear layer.
        """

        new_weight = torch.nn.parameter.Parameter(
            torch.cat([self.layer[depht]['L'].weight.detach(), weight_to_add * lambda_w], dim=1), requires_grad=True)
        new_bias = torch.nn.parameter.Parameter(self.layer[depht]['L'].bias.detach(), requires_grad=True)
        new_in_features, new_out_features = new_weight.shape[1], new_weight.shape[0]

        del self.layer[depht]['L']

        self.layer[depht]['L'] = torch.nn.Linear(new_in_features, new_out_features)

        self.layer[depht]['L'].weight = new_weight
        self.layer[depht]['L'].bias = new_bias

    def help_add_K_neurons_linear_left(self, depth, weight_to_add, bias_to_add, lambda_w=1.0):
        """
        Add the ingoing weights of the added neurons neurons for a linear layer.
        """
        new_weight = torch.nn.parameter.Parameter(torch.cat([self.layer[depth]['L'].weight, weight_to_add * lambda_w]),
                                                  requires_grad=True)
        new_bias = torch.nn.parameter.Parameter(torch.cat([self.layer[depth]['L'].bias, bias_to_add * lambda_w]),
                                                requires_grad=True)

        del self.layer[depth]['L']

        self.layer[depth]['L'] = torch.nn.Linear(new_weight.shape[1], new_weight.shape[0])

        self.layer[depth]['L'].weight = new_weight
        self.layer[depth]['L'].bias = new_bias

    def help_add_K_neurons_conv2d_right(self, depth, weight_to_add, lambda_w=1):
        """
        Add the outgoing weights of the added neurons neurons for a Conv2d layer.
        """
        my_strides, my_padding = self.layer[depth]['C'].stride, self.layer[depth]['C'].padding

        new_weight = torch.nn.parameter.Parameter(
            torch.cat([self.layer[depth]['C'].weight.detach(), weight_to_add * lambda_w], dim=1), requires_grad=True)
        new_bias = torch.nn.parameter.Parameter(self.layer[depth]['C'].bias.detach(), requires_grad=True)

        del self.layer[depth]['C']

        self.layer[depth]['C'] = torch.nn.Conv2d(new_weight.shape[1], new_weight.shape[0],
                                                 kernel_size=(new_weight.shape[2], new_weight.shape[3]),
                                                 stride=my_strides, padding=my_padding)

        self.layer[depth]['C'].weight = new_weight
        self.layer[depth]['C'].bias = new_bias

        # self.optimizer.add_param_group({'params' : self.layer[layer]['C'].weight})

    def help_add_K_neurons_conv2d_left(self, depth, weight_to_add, bias_to_add, lambda_w=1.0):
        """
        Add the ingoing weights of the added neurons neurons for a Conv2d layer.
        """
        my_kernel_size, my_stride, my_padding = self.layer[depth]['C'].kernel_size, self.layer[depth]['C'].stride, \
        self.layer[depth]['C'].padding

        new_weight = torch.nn.parameter.Parameter(
            torch.cat([self.layer[depth]['C'].weight.detach(), weight_to_add * lambda_w]), requires_grad=True)
        new_bias = torch.nn.parameter.Parameter(
            torch.cat([self.layer[depth]['C'].bias.detach(), bias_to_add * lambda_w]), requires_grad=True)

        del self.layer[depth]['C']

        self.layer[depth]['C'] = torch.nn.Conv2d(new_weight.shape[1], new_weight.shape[0], kernel_size=my_kernel_size,
                                                 stride=my_stride, padding=my_padding)

        self.layer[depth]['C'].weight = new_weight
        self.layer[depth]['C'].bias = new_bias

        autre_BatchNorm = torch.nn.BatchNorm2d(self.layer[depth]['C'].out_channels, device=my_device_0)
        autre_BatchNorm.weight = torch.nn.parameter.Parameter(torch.cat([self.layer[depth]['B'].weight,
                                                                         torch.ones(self.alpha.shape[0],
                                                                                    device=my_device_0)]))
        autre_BatchNorm.bias = torch.nn.parameter.Parameter(torch.cat([self.layer[depth]['B'].bias,
                                                                       torch.zeros(self.alpha.shape[0],
                                                                                   device=my_device_0)]))

        del self.layer[depth]['B']

        self.layer[depth]['B'] = autre_BatchNorm

    def add_K_neurons_linear_convolution(self, depth, new_weight_1, new_weight_2, bias_1, lambda_w=1.0):
        """
        Add the new neurons to self at depth.
        """
        if depth < self.break_conv:
            self.help_add_K_neurons_conv2d_left(depth, new_weight_1, bias_1, lambda_w=lambda_w)
        else:
            self.help_add_K_neurons_linear_left(depth, new_weight_1, bias_1, lambda_w=lambda_w)

        if depth + 1 < self.break_conv:
            self.help_add_K_neurons_conv2d_right(depth + 1, new_weight_2, lambda_w=1)
        else:
            self.help_add_K_neurons_linear_right(depth + 1, new_weight_2, lambda_w=1)

        self.update_architecture(depth, new_weight_1.shape, new_weight_2.shape)

        if not (getattr(self, 'dico_XTX_rk_Add', None) is None):
            self.update_XTX_rk_Add_apriori(depth)

    @torch.no_grad()
    def transform_dico_w(self, depth):
        """
        Concatenates zero vectors to the BestUpdate.
        
        The function is called after the adding of neurons when 
        the BestUpdate has been computed on the previous architecture.
        As its shape doesn't match the current architecture, 
        it is updated by adding zero vectors.
        """

        if self.dico_w is not None and self.layer[depth + 1][self.layer_name[depth + 1][0]].weight.shape != self.dico_w[
            'weight'].shape:
            alpha, omega, bias_alpha = self.alpha, self.omega, self.bias_alpha
            if self.alpha.shape[0] > 0 and self.layer_name[depth][0] == 'L':
                self.dico_w['weight'] = torch.cat([self.dico_w['weight'],
                                                   torch.zeros((self.dico_w['weight'].shape[0], alpha.shape[0]),
                                                               device=my_device)], dim=1)
            elif self.alpha.shape[0] > 0 and self.layer_name[depth][0] == 'C' and self.layer_name[depth + 1][0] == 'L':
                new_zero_size = self.outputs_size_after_activation[depth][0] * \
                                self.outputs_size_after_activation[depth][1]
                self.dico_w['weight'] = torch.cat([self.dico_w['weight'], torch.zeros(
                    (self.dico_w['weight'].shape[0], alpha.shape[0] * new_zero_size), device=my_device)], dim=1)
            elif self.alpha.shape[0] > 0 and self.layer_name[depth][0] == 'C' and self.layer_name[depth + 1][0] == 'C':
                new_zero_size = self.outputs_size_after_activation[depth][0] * \
                                self.outputs_size_after_activation[depth][1]
                self.dico_w['weight'] = torch.cat([self.dico_w['weight'], torch.zeros(omega.shape, device=my_device)],
                                                  dim=1)

    #####################################################
    ################## amplitude factor #################

    def compute_Loss_batch_NG(self, depth, fct_to_apply, ampl_fact=0., indx=None, reduction='mean'):
        """
        Computes the avearged and individual losses of the network for a minibatch 
        either when adding the BestUpdate or the Newneurons, 
        with the amplitude factor 'ampl_fact'.
        """
        if indx is None:
            indx = self.ind_lmbda
        L = torch.tensor(0., device=my_device)
        l = torch.zeros(indx.shape[0], device=my_device)
        for sous_indices in range(math.ceil(indx.shape[0] / self.max_batch_estimation)):
            sous_ind = indx[sous_indices * self.max_batch_estimation: (sous_indices + 1) * self.max_batch_estimation]
            sous_seed = self.seed_lmbda[
                        sous_indices * self.max_batch_estimation: (sous_indices + 1) * self.max_batch_estimation]
            X, Y = self.get_batch(indices=sous_ind, seed=sous_seed, device=my_device_0)
            l[sous_indices * self.max_batch_estimation: (sous_indices + 1) * self.max_batch_estimation] = self.Loss(
                fct_to_apply(X, depth=depth, lmbda=ampl_fact)[0], Y.to(my_device), reduction='none')
            L += l[sous_indices * self.max_batch_estimation: (sous_indices + 1) * self.max_batch_estimation].sum() / \
                 indx.shape[0]

        return L, l

    @torch.no_grad()
    def compute_decay_upgrade_glissant(self, depth_ajout, exp=2, method='Add'):
        """
        Evaluates the loss of the network for different values of the amplitude factor 
        and returns the one wich minimizes the avecarged loss.
        """
        if method == 'Add':
            fct_to_apply = self.forward_ajout
            self.alpha_omega_sign(depth_ajout)
            k_min = math.ceil(torch.log(self.init_deplacement) / np.log(exp)) - 1
        elif method == 'NG':
            fct_to_apply = self.forward_naturel
            k_min = math.ceil(torch.log(self.init_deplacement_NG) / np.log(exp)) - 1
        else:
            raise ValueError('method should be either Add or NG')

        k_max = math.ceil(np.log(self.max_amplitude) / np.log(exp))
        Loss_ = torch.zeros(k_max - k_min + 1, device=my_device)
        betas = torch.tensor([0] + [exp ** k for k in range(k_min, k_max)], device=my_device)
        portion_gain = torch.ones(betas.shape, device=my_device_0)
        portion_gain[0] = 1.
        i, condition = 0, True
        while i < len(betas) and condition:
            beta = betas[i]
            Loss_[i], pg = self.compute_Loss_batch_NG(depth_ajout, fct_to_apply, ampl_fact=beta)
            if i > 0:
                portion_gain[i] = ((pg - Loss_init_b) <= 0).sum() / self.ind_lmbda.shape[0]
            else:
                Loss_init_b = copy.deepcopy(pg)

            if i > 0 and Loss_[i] > Loss_[i - 1]:
                condition = False
                betas = betas[:i + 1]
                Loss_ = Loss_[:i + 1]
                portion_gain = portion_gain[:i + 1]

            else:
                i += 1
        print('Delta Loss : ', ((Loss_ - Loss_[0])[:3]).tolist(), ' ...', ((Loss_ - Loss_[0])[-3:]).tolist())
        print('betas : ', betas[:3].tolist(), '...', betas[-3:].tolist())
        beta_min, mon_accroissement, ma_portion_gain = UTILS.condition_valeurs_amplitude_factor(betas, Loss_,
                                                                                                self.accroissement_decay,
                                                                                                portion_gain=portion_gain)
        self.accroissement = mon_accroissement
        self.portion_gain = portion_gain
        self.Loss_gain = Loss_
        self.choosen_portion_gain = ma_portion_gain
        self.beta_min = beta_min
        return beta_min

    #####################################################
    ################## Training #########################

    def train_batch(self, nbr_epoch=10, limite_temps=None, optimizer=None, cte_epoch=0):
        """
        Trains the neural network for some epochs.
        """
        loss_list_train, loss_list_test, loss_list_valid = [], [], []
        accuracy_train, accuracy_test, accuracy_valid = [], [], []
        my_time, indices, indices_te, coef = [], None, None, 1

        tr_p, te_p = torch.randperm(self.len_train_dataset), torch.randperm(self.len_test_dataset)
        perm_train = torch.cat([tr_p, tr_p[:self.batch_size]])
        perm_test = torch.cat([te_p, te_p[:self.batch_size]])

        if optimizer is None:
            optimizer = self.optimizer
        if limite_temps is not None:
            epoch = 1000000

        t0 = time.time()
        criterion = self.Loss

        for e in range(1, math.ceil(nbr_epoch) + 1):
            if not (getattr(self, 'scheduler', None) is None) and e - 1 + cte_epoch >= 1:
                print('e + scheduler.step :', e)
                self.scheduler.step(e + cte_epoch)
            if e > 1:
                print('Acc test :', np.array(accuracy_test[-50:]).mean(),
                      'Acc train :', np.array(accuracy_train[-50:]).mean(),
                      'lr :', optimizer.param_groups[0]['lr'])
            if e - 1 < nbr_epoch < e:
                # if (e - 1) < nbr_epoch and e > nbr_epoch:
                coef = nbr_epoch - int(nbr_epoch)
                # print('coef :', coef)
            for t in range(int((self.len_train_dataset // self.batch_size) * coef) + 1):
                if getattr(self, 'tr_loader', None) is None:
                    indices = perm_train[t * self.batch_size: (t + 1) * self.batch_size]
                X_input, Y_input = self.get_batch(indices=indices, data='tr', device=my_device_0)

                optimizer.zero_grad()
                Y_pred = self(X_input)
                Loss = criterion(Y_pred, Y_input)
                Loss.backward()

                if not (self.gradient_clip is None):
                    # print("*** gradient clip : A DEFINIR ***")
                    # assert False
                    for p in self.parameters():
                        torch.nn.utils.clip_grad_value_(p, self.gradient_clip)

                optimizer.step()

                with torch.no_grad():
                    if getattr(self, 'tr_loader', None) is None:
                        indices_te = perm_test[t * self.batch_size: (t + 1) * self.batch_size]
                    X_te, Y_te = self.get_batch(data='te', indices=indices, device=my_device_0)
                    y_pred = self(X_te)
                    loss_list_test.append(criterion(y_pred, Y_te).item())
                    loss_list_train.append(Loss.detach().item())
                    # loss_list_valid.append(self.Loss(self(X_valid_rescale.to(my_device_0)), Y_valid_rescale.to(my_device_0)).item())
                    accuracy_test.append(UTILS.calculate_accuracy(y_pred, Y_te))
                    accuracy_train.append(UTILS.calculate_accuracy(self(X_input), Y_input))
                    # accuracy_valid.append(UTILS.calculate_accuracy(self(X_valid_rescale.to(my_device_0)), Y_valid_rescale.to(my_device_0)))
                    my_time.append(time.time() - t0)
                    if not (getattr(self, 'warmup_scheduler', None) is None) and e - 1 + cte_epoch < 1:
                        self.warmup_scheduler.step()

        return (np.array(loss_list_train),
                np.array(loss_list_test),
                np.array(loss_list_valid),
                np.array(accuracy_train),
                np.array(accuracy_test),
                np.array(accuracy_valid),
                np.array(my_time))

    def unfold_M(self, depth):
        """
        Unfold the activity of the network at depth.
        """
        kernel_size = self.layer[depth]['C'].kernel_size
        padding = self.layer[depth]['C'].padding
        stride = self.layer[depth]['C'].stride
        # unfold_f = torch.nn.Unfold(kernel_size, dilation=1, padding=padding, stride=stride)
        M = torch.nn.Unfold(kernel_size, dilation=1, padding=padding, stride=stride)(GLOBALS.activity[depth]).permute(0, 2, 1)

        if self.random_M:
            ## f-test ##
            return torch.randn_like(M)
        else:
            return M

    #####################################################
    ################## Hooks ############################

    def __activities_input(self, module, i, o):
        if self.random_activity:
            GLOBALS.activity[self.where] = torch.randn(i[0].shape, device=my_device_1)
        else:
            GLOBALS.activity[self.where] = i[0].detach().to(my_device_1)

    def __activities_output_with_gradient(self, module, i, o):
        if self.random_activity:
            GLOBALS.outputs[self.where] = torch.randn(o[0].shape, device=my_device_1)
        else:
            o.requires_grad = True
            GLOBALS.outputs[self.where] = o
        return o

    def register_activities(self, i0, i1):
        """
        Registers the activity of the network for all layers at depth between
        i0 and i1.
        """
        for f in range(max(i0, 1), min(i1 + 1, self.deep + 1)):
            if not (self.h.__contains__(f)):
                if self.layer_name[f][0] == 'C':
                    self.h[f] = self.layer[f]['C'].register_forward_hook(self.__activities_input)
                else:
                    self.h[f] = self.layer[f]['L'].register_forward_hook(self.__activities_input)

    def remove_activities(self, l=None):
        """
        Removes the registered hooks.
        """
        if l is None:
            l = self.h.keys()
        for key in l:
            self.h[key].remove()
        self.h.clear()

    def __BestUpdateHook(self, module, i, o):
        def onelayerforward(x, w, b, padding=None, stride=None):
            if self.layer_name[self.where][0] == 'C':
                return self.f_conv(x, w, b, padding=padding, stride=stride)
            else:
                return self.f_lin(x, w, lmbda * b)

        lmbda = self.ampl_fact_temp
        padding = getattr(module, 'padding', None)
        stride = getattr(module, 'stride', None)
        w = self.dico_w['weight']
        b = self.dico_w['bias']

        DE = onelayerforward(i[0], w, b, padding=padding, stride=stride)
        self.norm_x_over_norm_DE = torch.linalg.norm(o) / (torch.linalg.norm(DE) + 1e-8)
        return o + lmbda * DE

    def AddHookleft(self, module, i, o):
        def onelayerforward(x, w, b, padding=None, stride=None):
            if self.layer_name[self.where][0] == 'C':
                return self.f_conv(x, w, b, padding=padding, stride=stride)
            else:
                return self.f_lin(x, w, b)

        lmbda = np.sqrt(self.ampl_fact_temp)
        padding = getattr(module, 'padding', None)
        stride = getattr(module, 'stride', None)
        w = self.alpha
        b = self.bias_alpha

        DE = onelayerforward(i[0], w, b, padding=padding, stride=stride)
        self.norm_x_over_norm_DE_left = torch.linalg.norm(o) / (torch.linalg.norm(DE) + 1e-8)
        self.lmbdaDE = lmbda * DE
        self.output = o
        self.BN_lmbdaDE = self.lmbdaDE

    def AddHookright(self, module, i, o):
        def onelayerforward(x, w, b, padding=None, stride=None):
            if self.layer_name[self.where][0] == 'C':
                return (self.f_conv(x, w, b, padding=padding, stride=stride))
            else:
                return (self.f_lin(x, w, b))

        lmbda = np.sqrt(self.ampl_fact_temp)
        padding = getattr(module, 'padding', None)
        stride = getattr(module, 'stride', None)
        w = self.omega
        b = torch.nn.parameter.Parameter(torch.zeros(self.omega.shape[0], device=my_device_1))

        DE = onelayerforward(self.fct_BN_lmbdaDE, w, b, padding=padding, stride=stride)
        if 'B' in self.layer_name[self.where - 1]:
            self.norm_x_over_norm_DE_right = torch.linalg.norm(o) / (torch.linalg.norm(DE) + 1e-8)
        else:
            self.norm_x_over_norm_DE_right = torch.linalg.norm(o) / (torch.linalg.norm(DE) / lmbda + 1e-8)
        return (o + lmbda * DE)

    def AddHookfct(self, module, i, o):
        self.fct_BN_lmbdaDE = self.copy_fct(self.BN_lmbdaDE)
        if self.where == self.break_conv - 1:
            # print('FLATTEN')
            self.fct_BN_lmbdaDE = self.fct_BN_lmbdaDE.flatten(start_dim=1)

    def AddHookBN(self, module, i, o):
        extented_BN = torch.nn.BatchNorm2d(module.num_features + self.alpha.shape[0], device=my_device_1)
        extented_BN.weight = torch.nn.parameter.Parameter(torch.cat([module.weight,
                                                                     torch.ones(self.alpha.shape[0],
                                                                                device=my_device_0)]))
        extented_BN.bias = torch.nn.parameter.Parameter(torch.cat([module.bias,
                                                                   torch.zeros(self.alpha.shape[0],
                                                                               device=my_device_0)]))
        self.BN_lmbdaDE = extented_BN(torch.concatenate([o, self.lmbdaDE], dim=1))[:, -self.alpha.shape[0]:]

    #####################################################
    ################## F test ###########################

    def update_XTX_rk_Add_apriori(self, depth_add):
        if depth_add < self.deep - 1:
            coef = self.nbr_added_neuron / self.nbr_old_neuron
            plus_size = int(self.dico_XTX_rk_Add[depth_add + 1]['size'] * coef)

            print('plus_size :', plus_size)
            self.dico_XTX_rk_Add[depth_add + 1]['size'] += plus_size
            self.dico_XTX_rk_Add[depth_add + 1]['rank'] += int(coef * self.dico_XTX_rk_Add[depth_add + 1]['rank'])
            self.dico_XTX_rk_Add[depth_add + 1]['rank_default'] += int(
                coef * self.dico_XTX_rk_Add[depth_add + 1]['rank_default'])
            self.dico_XTX_rk_Add[depth_add + 1]['batch_size'] += int(
                coef * self.dico_XTX_rk_Add[depth_add + 1]['batch_size'])
