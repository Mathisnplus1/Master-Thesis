###############################
# Compute the best update and #
# the new neurons according   # 
# to EB definition            #
###############################


from typing import Iterable, Optional, Any
import math
import numpy as np
import torch
import copy
from define_devices import my_device_0, my_device_1

import UTILS
import T_S_F_N
import GLOBALS
from TINY import TINY


###############################################
############ Matrices MSM and MDV #############

def __batch_MSM_MDV(model,
                    num: int,
                    activity_b: torch.Tensor,
                    dv: torch.Tensor,
                    tensor_s: torch.Tensor,
                    tensor_n: torch.Tensor,
                    depth: int,
                    method='NG'):
    # frac = 1
    if num == 0:
        ### NG or Add L-L
        # tensor_s += b b^T
        # tensor_n += dv b
        if len(dv.shape) > 2:
            tensor_s.add_(torch.einsum('ijk, ijl->kl', activity_b, activity_b))
            n_batch, n_patch, patch_size = activity_b.shape
            assert dv.shape[0] == n_batch, \
                f"dv.shape[0] should be {n_batch} but is {dv.shape[0]}"
            assert dv.shape[2] == n_patch, \
                f"dv.shape[2] should be {n_patch} but is {dv.shape[2]}"
            tensor_n.add_(torch.einsum('ijk, ilj->kl', activity_b, dv))
        else:
            tensor_s.add_(torch.einsum('ik, ij->kj', activity_b, activity_b))
            tensor_n.add_(torch.einsum('ik, ij->kj', activity_b, dv))
    elif num == 1 or num == 2:
        if GLOBALS.dico_F.get(depth, None) is not None:
            if torch.get_device(activity_b) == -1:
                fx = torch.matmul(GLOBALS.dico_F[depth].to_dense(), activity_b)
            else:
                # fx = torch.matmul(GLOBALS.dico_F[depth], activity_b)
                # TODO: fix "RuntimeError: expand is unsupported for Sparse tensors"
                fx = torch.matmul(GLOBALS.dico_F[depth].to_dense(), activity_b)
        else:
            fx = activity_b
        # WHAT is fx ?

        if num == 1:
            ### Add C-C
            tensor_s.add_(T_S_F_N.compute_S_for_ConvConv(FM=fx, tt=GLOBALS.dico_tt[depth]), alpha=activity_b.shape[0])
            tensor_n.add_((T_S_F_N.compute_N_for_ConvConv(FM=fx, tensor_t=GLOBALS.dico_mask_tensor_t[depth], dv=dv)).T)
            # TODO: WHY is there no alpha=activity_b.shape[0] here?

        elif num == 2:
            ### Add C - L
            tensor_s.add_(T_S_F_N.compute_S_for_ConvLin(fx), alpha=activity_b.shape[0])
            tensor_n.add_(T_S_F_N.compute_N_for_ConvLin(dv, fx).T, alpha=activity_b.shape[0])
    else:
        raise ValueError(f"num should be 0, 1 or 2 but is {num}")


def __MSM_MDV_shapes(model: TINY,
                     depth: int,
                     method: str
                     ) -> tuple[int, int]:
    layer_weight_shape = model.layer[depth][model.layer_name[depth][0]].weight.shape
    if method == 'NG':
        layer_weightp1_shape = model.layer[depth][model.layer_name[depth][0]].weight.shape
    elif method == 'Add':
        layer_weightp1_shape = model.layer[depth + 1][model.layer_name[depth + 1][0]].weight.shape
    else:
        raise ValueError(f"method should be either NG or Add but is {method}")

    if model.layer_name[depth][0] == 'C':
        MSM_shape = layer_weight_shape[1] * layer_weight_shape[-1] * layer_weight_shape[-2] + 1
        if len(layer_weightp1_shape) > 2:
            conv_multiplier = (method == 'Add') * layer_weightp1_shape[-1] * layer_weightp1_shape[-2] + (method == 'NG')
            MDV_shape = layer_weightp1_shape[0] * conv_multiplier
        elif len(layer_weightp1_shape) == 2:
            MDV_shape = model.layer[depth + 1]['L'].out_features * model.outputs_size_after_activation[depth][0] * \
                        model.outputs_size_after_activation[depth][1]
        else:
            raise ValueError(f"len(layer_weightp1_shape) should be >= 2 but is {len(layer_weightp1_shape)}")
    elif model.layer_name[depth][0] == 'L':
        MSM_shape = layer_weight_shape[1] + 1
        MDV_shape = layer_weightp1_shape[0]
    else:
        raise ValueError(f"layer_name[depth][0] should be either L or C but is {model.layer_name[depth][0]}")

    return MSM_shape, MDV_shape


def __MSM_MDV(model: TINY,
              depth: int,
              method: str = 'NG'):
    indices = model.ind

    tensor_s_shape, tensor_n_shape = __MSM_MDV_shapes(model, depth, method)
    tensor_n = torch.zeros((tensor_s_shape, tensor_n_shape), device=my_device_0)
    # tensor_n <-> N = (1/n) B[l] (V_goal[l]_proj)^T
    tensor_s = torch.zeros((tensor_s_shape, tensor_s_shape), device=my_device_0)
    # tensor_s <-> S_l = (1/n) B[l] B[l]^T

    for i in range(math.ceil(indices.shape[0] / model.max_batch_estimation)):
        sous_indices = indices[i * model.max_batch_estimation: (i + 1) * model.max_batch_estimation]
        sous_seed = model.seed[i * model.max_batch_estimation: (i + 1) * model.max_batch_estimation]
        input_x, target_y = model.get_batch(device=my_device_0, indices=sous_indices, seed=sous_seed)
        # frac = 1
        GLOBALS.print_globals()
        if method == 'NG':
            dv = -model.deplacement_voulu(depth, X=input_x, Y=target_y).to(my_device_1)
            # dv = V_goal[l - 1]
        elif method == 'Add':
            dv = -model.deplacement_voulu(depth + 1, X=input_x, Y=target_y).to(my_device_1)
            # dv = V_goal[l]
            # TODO: solve padding issue
            m = UTILS.layer_w_0_star(model.dico_w,
                                     padding=model.skeleton[depth].get('padding', 1))
            # m = dW[l]*
            UTILS.DV_proj(dv,
                          GLOBALS.activity.get(depth + 1, None),
                          m,
                          architecture_growth=model.architecture_growth)
            # dv = V_goal[l]_proj = V_goal[l] - dW[l]* B[l-1]
        else:
            raise ValueError(f"method should be either NG or Add but is {method}")

        del input_x, target_y

        # TODO: fix the fact that if method is 'NG' dv = V_goal[l - 1] and if method is 'Add' dv = V_goal[l]_proj
        GLOBALS.print_globals()
        if model.layer_name[depth] == 'L':
            b = torch.cat([GLOBALS.activity[depth],
                           torch.ones((GLOBALS.activity[depth].shape[0], 1), device=my_device_1)], dim=1)
            # X = B[l-1]
            __batch_MSM_MDV(None, 0, activity_b=b, dv=dv, tensor_s=tensor_s, tensor_n=tensor_n, depth=depth, method=method)
            # tensor_s += B[l-1] B[l-1]^T
            # tensor_n += B[l-1] V_goal[l]_proj if method == 'Add' else B[l-1] V_goal[l - 1]

        elif model.layer_name[depth][0] == 'C':
            b = model.unfold_M(depth)
            # M = B[l-1] in (n, S(d), W(d), H(d)) unfolded into (N, L, S(d) * W_d * H_d)
            b = torch.cat([b, torch.ones((b.shape[0], b.shape[1], 1), device=my_device_1)], dim=2)

            if method == 'NG':
                # B[depth] : GLOBALS.activity[depth] in (n, S(d), W(d), H(d)) unfolded into ???

                # X = B[l-1] with the bias (?)
                dv = dv.flatten(start_dim=2)
                # dv = V_goal[depth] in (n, S(d), W(d), H(d)) flattened into (n, S(d), W(d) * H(d))
                __batch_MSM_MDV(None, 0, activity_b=b, dv=dv, tensor_s=tensor_s,
                                tensor_n=tensor_n, depth=depth, method=method)
                # tensor_s += B[l-1] B[l-1]^T
                # tensor_n += B[l-1] V_goal[l]
            elif method == 'Add':
                if model.layer_name[depth + 1][0] == 'C':
                    # num = 1
                    __batch_MSM_MDV(None, 1, activity_b=b, dv=dv, tensor_s=tensor_s,
                                    tensor_n=tensor_n, depth=depth, method=method)
                elif model.layer_name[depth + 1][0] == 'L':
                    # num = 2
                    __batch_MSM_MDV(None, 2, activity_b=b, dv=dv, tensor_s=tensor_s,
                                    tensor_n=tensor_n, depth=depth, method=method)
                else:
                    raise ValueError(f"layer_name[depth + 1] should be either L or C but is {model.layer_name[depth + 1]}")
            else:
                raise ValueError(f"method should be either NG or Add but is {method}")
        else:
            raise ValueError(f"layer_name[depth] should be either L or C but is {model.layer_name[depth]}")

        try:  # WHY IS DEL USED HERE?
            del dv
            del b
        except UnboundLocalError:
            pass

    tensor_s /= model.ind.shape[0]
    tensor_n /= model.ind.shape[0]

    return tensor_s, tensor_n


###############################################
############ Compute BestUpdate  ##############
############ Compute NewNeurons  ##############
###############################################

def check_settings(model, depth, method):
    if model.layer_name[depth][0] == 'C':
        assert model.layer[depth]['C'].kernel_size == (3, 3)
        assert model.layer[depth]['C'].padding == 'same' or model.layer[depth]['C'].padding == (1, 1)
        assert model.layer[depth]['C'].stride == (1, 1)

    if method == 'Add' and model.layer_name[depth + 1][0] == 'C':
        assert model.layer[depth + 1]['C'].kernel_size == (3, 3)
        assert model.layer[depth + 1]['C'].padding == 'same' or model.layer[depth]['C'].padding == (1, 1)
        assert model.layer[depth + 1]['C'].stride == (1, 1)


def compute_NG(model: TINY, depth: int, update=True, compute_gain=True) -> None:
    # check_settings(model, depth, 'NG')
    print('\n')
    print('*** started ', 'NG', 'at ', depth, '***')
    print('Batch size for estimation :', model.ind.shape[0])

    model.portion_gain = torch.ones(1, device=my_device_0)
    model.Loss_gain = torch.ones(1, device=my_device_0)
    model.choosen_portion_gain = torch.ones(1, device=my_device_0)

    model.accroissement = torch.tensor(0., device=my_device_0)
    method = 'NG'

    if model.dico_w is None:
        model.register_activities(depth, depth)

        MSM, MDV = __MSM_MDV(model, depth, method=method)
        try:
            inverse = torch.linalg.inv(MSM)
        except torch.linalg.LinAlgError:
            try:
                print('*** SINGULAR MATRIX ***')
                inverse = torch.linalg.pinv(MSM)
        
            except torch.linalg.LinAlgError:
                print('*** MSM := Id ***')
                inverse = torch.eye(MSM.shape[0], device=my_device_0)

        delta_w = torch.matmul(inverse, MDV)

        model.dico_w = {'weight': delta_w.permute(1, 0)[:, :-1].reshape(
            model.layer[depth][model.layer_name[depth][0]].weight.shape).to(my_device_0),
                        'bias': delta_w.permute(1, 0)[:, -1].to(my_device_0)}
        model.remove_activities()
    GLOBALS.activity.clear()

    # t_lambda_NG = time.time()
    # lmbda = 0.
    if compute_gain:
        model.selection_NG(model, depth)
        if not (model.dico_w is None):
            if model.lambda_method_NG > 0:
                lmbda = model.lambda_method_NG.clone()
            else:
                model.rescale_dico_w(depth)
                lmbda = model.compute_decay_upgrade_glissant(depth, exp=model.exp, method='NG')

            # t_lambda_NG = time.time() - t_lambda_NG
            if lmbda > 0 and update:
                print('*** updated layer :', depth, '***')
                model.gradient_naturel(depth, model.dico_w, lmbda=lmbda)
                model.updated_NG = True

    GLOBALS.activity.clear()


def add_neurons(model: TINY,
                depth: int,
                variable_names=['alpha'],
                alpha=None,
                omega=None,
                bias_alpha=None,
                valeurs_propres=[0],
                update=True,
                M=None,
                MDV=None,
                compute_gain: bool=True
                ) -> None:
    # check_settings(model, depth, 'Add')
    print('\n')
    print('*** started ', 'Add', 'at ', depth, '***')
    print('Batch size for estimation :', model.ind.shape[0])
    model.portion_gain = torch.ones(1, device=my_device_0)
    model.Loss_gain = torch.ones(1, device=my_device_0)
    model.choosen_portion_gain = torch.ones(1, device=my_device_0)

    method = 'Add'
    model.nbr_added_neuron = 0
    model.nbr_old_neuron = model.layer[depth][model.layer_name[depth][0]].weight.shape[0]
    model.amplitude_factor.mul_(0.)
    model.accroissement.mul_(0.)
    model.accroissement_NG.mul_(0.)

    neurontype = model.layer_name[depth][0] + model.layer_name[depth + 1][0]

    if alpha is None or omega is None:
        model.register_activities(depth, depth + 1)
        MMT, MDV = __MSM_MDV(model, depth, method=method)
        model.remove_activities()

        eigen_matrix_u, eigen_values_sigma, S_1demiN, _ = UTILS.S_1demiN(
            M=MMT, MDV=MDV, MDV_vrai_gaus=None, architecture_growth=model.architecture_growth)
        alpha_computed, omega_computed, eigen_values_computed = UTILS.SVD_Smoins1demiN(S_1demiN=S_1demiN,
                                                                                       P=eigen_matrix_u,
                                                                                       D=eigen_values_sigma)

        model.alpha_computed = alpha_computed.to(my_device_0)
        model.omega_computed = omega_computed.to(my_device_0)
        model.valeurs_propres_computed = eigen_values_computed.to(my_device_0)

        model.alpha = model.alpha_computed.clone()
        model.omega = model.omega_computed.clone()
        model.valeurs_propres = model.valeurs_propres_computed.clone()

        model.alpha, model.bias_alpha, model.omega = UTILS.reshape_neurons(model.alpha, model.omega, model.skeleton,
                                                                           depth, neurontype=neurontype)
    else:
        model.alpha, model.omega, model.bias_alpha, model.valeurs_propres = alpha, omega, bias_alpha, valeurs_propres

    GLOBALS.activity.clear()
    model.amplitude_factor *= 0.

    if compute_gain:
        model.selection_neuron(model, depth)
        if not (model.alpha is None):
            model.rescale_alpha_omega(depth)
            if model.lambda_method > 0:
                lambda_w = model.lambda_method.clone()
            else:
                lambda_w = model.compute_decay_upgrade_glissant(depth, exp=model.exp, method='Add')
            model.amplitude_factor = lambda_w.clone()
            if lambda_w > 0. and update:
                model.nbr_added_neuron = model.alpha.shape[0]
                model.scale_new_neurons(model.amplitude_factor)
                model.add_K_neurons_linear_convolution(depth, model.alpha, model.omega, model.bias_alpha, lambda_w=1.)
                print('*** Added neurons at ', depth, '***')
                print('\n')


###############################################
############## eval EB ########################


def eval_EB_at_depth(model: TINY, depth: int) -> dict[str, Any]:
    model.ind, model.ind_lmbda, model.seed, model.seed_lmbda = None, None, None, None
    model.dico_w = None
    model.how_to_define_batchsize(model, depth + 1, method='NG')
    compute_NG(model=model, depth=depth + 1, update=False, compute_gain=False)

    model.ind, model.ind_lmbda, model.seed, model.seed_lmbda = None, None, None, None
    model.how_to_define_batchsize(model, depth, method='Add')
    add_neurons(model, depth, update=False, compute_gain=True)
    accroissement = model.accroissement.item()
    portion_gain = model.choosen_portion_gain.item()

    return {'depth': depth,
            'accroissement': accroissement,  # (Loss(t + 1) - Loss(t)) / Loss(t)
            'portion_gain': portion_gain,  # card { x | Loss(x, t + 1) < Loss(x, t) } / card { x }
            'alpha': copy.deepcopy(model.alpha),
            'omega': copy.deepcopy(model.omega),
            'bias_alpha': copy.deepcopy(model.bias_alpha),
            'vps': copy.deepcopy(model.valeurs_propres),
            'beta_min': model.beta_min,  # gamma : amplitude factor
            'dico_w': copy.deepcopy(model.dico_w)}


def eval_EB_at_depths(model: TINY,
                      depths: Optional[Iterable[int]] = None,
                      selection_criterion: str='accroissement'
                      ) -> tuple[dict[int, float], dict[int, dict[str, Any]]]:
    if depths is None:
        depths = list(range(1, model.deep))

    dico_expr_bottleneck = dict()
    dico_selection_criterion = dict()
    for depth in depths:
        dico_expr_bottleneck[depth] = eval_EB_at_depth(model, depth)
        dico_selection_criterion[depth] = dico_expr_bottleneck[depth][selection_criterion]
    return dico_selection_criterion, dico_expr_bottleneck


def where_is_EB_best_solved(model: TINY,
                            depths: Optional[Iterable[int]] = None
                            ) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    dico_selection_criterion, dico_expr_bottleneck = eval_EB_at_depths(model, depths=depths)
    values = np.array(list(dico_selection_criterion.values()))
    key = np.array(list(dico_selection_criterion.keys()))
    depth_in_decreasing_criterion = np.flip(key[values.argsort()])
    return depth_in_decreasing_criterion, dico_expr_bottleneck

###############################################
###############################################
