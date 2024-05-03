import torch
import copy
from define_devices import my_device, my_device_1


#####################################################
################## Solving EB #######################


def compute_S_for_ConvLin(M_1):
    """
    Compute the matrix S of proposition 3.2 when adding neurons
    between a convolution and a linear layer

    Return
    R^(j k) =  M^(n i j) M_(n i k) / n
    """
    return torch.einsum('nij, nik->jk', M_1, M_1) / M_1.shape[0]


def compute_N_for_ConvLin(DV, M):
    """
    Compute the matrix N of proposition 3.2 when adding neurons
    between a convolution and a linear layer
    """
    return (torch.einsum('nij, nk->ikj', M, DV) / M.shape[0]).flatten(end_dim=1)


def compute_N_for_ConvConv(FM: torch.Tensor,
                           tensor_t: torch.Tensor,
                           dv: torch.Tensor
                           ) -> torch.Tensor:
    """
    Compute the matrix N of proposition 3.2 when adding neurons
    between two convolutional layers.
    """
    _, _, j = tensor_t.shape
    assert FM.shape[1] == j, f"{FM.shape[1]=} {tensor_t.shape[2]=} should be equal"
    tfm = torch.einsum('ikj, mjl->ikml', tensor_t.to_dense(), FM)

    dv = dv.flatten(start_dim=2)
    k, _, i = dv.shape
    assert tfm.shape[0] == i, f"{tfm.shape[0]=} {dv.shape[2]=} should be equal"
    assert tfm.shape[2] == k, f"{tfm.shape[2]=} {dv.shape[0]=} should be equal"
    tfm_dv = torch.einsum('ijkl,kmi->mjl', tfm, dv).flatten(end_dim=1)
    return tfm_dv


def compute_S_for_ConvConv(FM: torch.Tensor, tt: torch.Tensor) -> torch.Tensor:
    """
    Compute the matrix S of proposition 3.2 when adding neurons
    between two convolutional layers.
    """
    # FM <-> activity_b
    # tt <-> (T(j)T(j))^T
    # print(f"{FM.shape=}, {S.shape=}")
    assert tt.shape[1] == FM.shape[1], f"{tt.shape[1]=} {FM.shape[1]=} should be equal (k)"
    SFM = torch.einsum('ik,lkj->ilj', tt.to_dense(), FM)
    assert SFM.shape[0] == FM.shape[1], f"{SFM.shape[0]=} {FM.shape[1]=} should be equal (i)"
    assert SFM.shape[1] == FM.shape[0], f"{SFM.shape[1]=} {FM.shape[0]=} should be equal (k)"
    MSM = torch.einsum('ikj, kim->jm', SFM, FM)
    MSM /= FM.shape[0]
    return MSM


#####################################################    
########## Constuction of T, F and S ################

def compute_mask_tensor_t_v2(input_shape: tuple[int, int],
                             conv: torch.nn.Conv2d
                             ) -> torch.Tensor:
    """
    Compute the tensor T
    For:
    - input tensor: B[-1] in (S[-1], H[-1]W[-1]) and (S[-1], H'[-1]W'[-1]) after the pooling
    - output tensor: B in (S, HW)
    - conv kernel tensor: W in (S, S[-1], Hd, Wd)
    T is the tensor in (HW, HdWd, H'[-1]W'[-1]) such that:
    B = W T B[-1]

    Parameters
    ----------
    input_shape: tuple
        shape of the input tensor B[-1] AFTER THE POOLING without the number of channels
        (H'[-1], W'[-1])
    conv: torch.nn.Conv2d
        convolutional layer applied to the input tensor B[-1]

    Returns
    -------
    tensor_t: torch.Tensor
        tensor T in (HW, HdWd, H[-1]W[-1])
    """
    with torch.no_grad():
        out_shape = conv(torch.empty((1, conv.in_channels, input_shape[0], input_shape[1]),
                         device=conv.weight.device)).shape[2:]

    tensor_t = torch.zeros(
        (out_shape[0] * out_shape[1], conv.kernel_size[0] * conv.kernel_size[1], input_shape[0] * input_shape[1]))
    unfold = torch.nn.Unfold(kernel_size=conv.kernel_size, padding=conv.padding, stride=conv.stride,
                             dilation=conv.dilation)
    t_info = unfold(
        torch.arange(1, input_shape[0] * input_shape[1] + 1).float().reshape((1, input_shape[0], input_shape[1]))).int()
    for lc in range(out_shape[0] * out_shape[1]):
        for k in range(conv.kernel_size[0] * conv.kernel_size[1]):
            if t_info[k, lc] > 0:
                tensor_t[lc, k, t_info[k, lc] - 1] = 1
    return tensor_t


def compute_mask_tensor_t(model, depth):
    # print('padding :', model.layer[depth]['C'].padding)
    print('Computing matrix T at depth : ', depth, ' ...')
    # T_tot = torch.tensor([])
    if model.layer[depth]['C'].padding == (0, 0):
        if hasattr(model.fct[depth], 'kernel_size'):
            # a = int(
            #     (model.outputs_size_after_activation[depth - 1][0] - model.skeleton[depth]['kernel_size'][0] + 1) / 2)
            # b = int(
            #     (model.outputs_size_after_activation[depth - 1][1] - model.skeleton[depth]['kernel_size'][1] + 1) / 2)
            # c = model.skeleton[depth]['kernel_size'][0]
            # d = model.skeleton[depth]['kernel_size'][1]

            return torch.tensor([]).to_sparse(3).to(my_device_1)
        else:
            a = int((model.outputs_size_after_activation[depth - 1][0] - model.skeleton[depth]['kernel_size'][0] + 1))
            b = int((model.outputs_size_after_activation[depth - 1][1] - model.skeleton[depth]['kernel_size'][1] + 1))
            c = model.skeleton[depth]['kernel_size'][0]
            d = model.skeleton[depth]['kernel_size'][1]
            T_0 = torch.zeros((c * d, a * b))

            for j in range(c):
                T_0[j * d: (j + 1) * d, j * (a):j * (a) + d] = torch.eye(d)

            T_tot = torch.tensor([])
            for i in range(1, (a - c + 1) * (b - c + 1) + 1):
                T_tot = torch.cat([T_tot, copy.deepcopy(torch.unsqueeze(T_0, dim=0))], dim=0)
                if int(i / (b - c + 1)) * (b - c + 1) == i:

                    T_0 = torch.cat([T_0[:, -(c):], T_0[:, :-(c)]], dim=1)
                else:
                    T_0 = torch.cat([T_0[:, -1:], T_0[:, :-1]], dim=1)

            return T_tot.to_sparse(3).to(my_device_1)

    else:
        if hasattr(model.fct[depth], 'kernel_size'):  # if the layer is a MaxPoolRelu
            a_old = int((model.outputs_size_after_activation[depth - 1][0]) / 2)
            b_old = int((model.outputs_size_after_activation[depth - 1][1]) / 2)
            # c = model.skeleton[depth]['kernel_size'][0]
            # d = model.skeleton[depth]['kernel_size'][1]
        else:  # if there is no MaxPoolRelu
            a_old = int((model.outputs_size_after_activation[depth - 1][0]))
            b_old = int((model.outputs_size_after_activation[depth - 1][1]))
            # c = model.skeleton[depth]['kernel_size'][0]
            # d = model.skeleton[depth]['kernel_size'][1]

        a, b, _ = model.outputs_size_after_activation[depth]  # TODO: check if this true in ConvLin

        assert a == a_old, f"{a=} {a_old=} should be equal (to fix this assert just define a = a_old)"
        assert b == b_old, f"{b=} {b_old=} should be equal (to fix this assert just define b = b_old)"

        old_fake_Malpha = torch.cat([torch.zeros((a, 1)),
                                 torch.arange(1, a * b + 1).reshape((a, b)),
                                 torch.zeros((a, 1))], dim=1)
        old_fake_Malpha = torch.cat([torch.zeros((1, old_fake_Malpha.shape[1])),
                                old_fake_Malpha,
                                torch.zeros((1, old_fake_Malpha.shape[1]))], dim=0)

        fake_Malpha = torch.zeros((a + 2, b + 2))  # TODO: maybe change the 2 with 2 * padding
        fake_Malpha[1:-1, 1:-1] = torch.arange(1, a * b + 1).reshape((a, b))
        # TODO: check if this is always true
        assert torch.allclose(fake_Malpha, old_fake_Malpha), 'fake_Malpha is not as expected'

        T_tot = torch.tensor([])
        s1, s2 = fake_Malpha.shape
        k1, k2 = model.layer[depth]['C'].kernel_size
        j = 0
        while j < (s1 * s2):  # for j in range(s1 * s2):
            if j // s1 + k2 <= a + 2 and j % s2 + k1 <= b + 2:
                # TODO: check that this assert is equivalent to the condition in the if
                assert fake_Malpha[j // s1: j // s1 + k2, j % s2: j % s2 + k1].numel() == k1 * k2, \
                    f"{fake_Malpha[j // s1: j // s1 + k2, j % s2: j % s2 + k1].numel()=} != {k1 * k2=}"

                # print(j//s1, j//s1 + k2, j%s2, j%s2 + k1)
                # T_save = torch.zeros(((k1 * k2, a * b)))
                save = fake_Malpha[j // s1: j // s1 + k2, j % s2: j % s2 + k1].flatten()
                index_nonzero = torch.nonzero((save > 0).float())
                i = torch.cat([index_nonzero, (save[save > 0] - 1).int().unsqueeze(dim=0).T], dim=1).T
                v = torch.ones(i.shape[1])

                tensor_t_save = torch.torch.sparse_coo_tensor(i, v, (k1 * k2, a * b)).to_dense()
                T_tot = torch.cat([T_tot, tensor_t_save.unsqueeze(dim=0)], dim=0)

            j += 1

        t2 = compute_mask_tensor_t_v2((a, b),
                                       model.layer[depth]['C'])
        assert torch.allclose(T_tot, t2), 'T_tot is not as expected'

        return T_tot.to_sparse(3).to(my_device_1)
    # del T_tot
    # gc.collect()
    # torch.cuda.empty_cache()


def compute_tensor_tt(model, depth, tensor_t):
    tensor_tt = torch.sparse.sum(
        torch.cat(
            [torch.unsqueeze(torch.sparse.mm(tensor_t[k].transpose(1, 0), tensor_t[k]), dim=0)
             for k in range(tensor_t.shape[0])]
        ), dim=0)

    tensor_t_dense = tensor_t.to_dense()
    tensor_tt_v2 = torch.einsum('ijk, ljk->il', tensor_t_dense, tensor_t_dense)
    assert torch.allclose(tensor_tt.to_dense(), tensor_tt_v2), 'tensor_tt is not as expected'

    return tensor_tt.to(my_device_1)  # TT := einsum('ijk, ljk->il', T, T)
    # del T_T_T
    # torch.cuda.empty_cache()
    # gc.collect()


def creation_T_C_pour_BCR(model, depth):
    if hasattr(model.fct[depth], 'kernel_size'):  # if the layer is a MaxPool

        T_tot = torch.tensor([], device=my_device)
        kernel_s = model.skeleton[depth]['kernel_size']
        if model.layer[depth]['C'].padding == (0, 0):
            a = model.outputs_size_after_activation[depth - 1][0] - model.skeleton[depth]['kernel_size'][0] + 1
            b = model.outputs_size_after_activation[depth - 1][1] - model.skeleton[depth]['kernel_size'][1] + 1
            T_0 = torch.zeros(
                (torch.tensor(model.outputs_size_after_activation[depth - 1][:2]) - kernel_s[0] + 1).prod(),
                device=my_device)

        else:
            a = model.outputs_size_after_activation[depth - 1][0]
            b = model.outputs_size_after_activation[depth - 1][1]
            T_0 = torch.zeros((torch.tensor(model.outputs_size_after_activation[depth - 1][:2])).prod(),
                              device=my_device)  # torch.zeros((a * b), device=my_device)

        for j in range(2):  # model.fct[depth].kernel_size
            T_0[j * a: j * a + 2] = 1. * torch.ones(2, device=my_device)

        if int(a * b / 4.) * 4 == a * b:
            for i in range(1, int((a) * (b) / 4.) + 1):
                T_tot = torch.cat([T_tot, copy.deepcopy(torch.unsqueeze(T_0, dim=0))], dim=0)
                if int((2 * i) / a) * a == 2 * i:
                    T_0 = torch.cat([T_0[-(2 + a):], T_0[:-(2 + a)]], dim=0)
                else:
                    T_0 = torch.cat([T_0[-2:], T_0[:-2]])
        else:

            for i in range(1, int((a - 1) * (b - 1) / 4) + 1):
                T_tot = torch.cat([T_tot, copy.deepcopy(torch.unsqueeze(T_0, dim=0))], dim=0)
                if int((2 * i) / (a - 1)) * (a - 1) == 2 * i:
                    T_0 = torch.cat([T_0[-(2 + a + 1):], T_0[:-(2 + a + 1)]], dim=0)
                else:
                    T_0 = torch.cat([T_0[-2:], T_0[:-2]])

        return (T_tot / (model.fct[depth].kernel_size ** 2)).to_sparse().to(my_device_1)

        # del T_tot
        # gc.collect()
        # torch.cuda.empty_cache()
