import sys
sys.path.append('/home/tau/mverbock/repot_git/code_expressivity_bott/2_GPUs')
import mes_imports as mi
import GLOBALS as G


############# TRAINING PARAMETERS ############

#nbr_epochs_betw_adding = G.nbr_epochs_betw_adding

############# SKIPCONNECTIONS ################

class SkipConnection_CB(mi.torch.nn.Module) :

    def __init__(self, in_channels, out_channels, stride = 1, padding = 'same', kernel_size = 1, bias = True):
        super(SkipConnection_CB, self).__init__()
        self.Conv = mi.torch.nn.Conv2d(in_channels, out_channels, stride = stride, padding = padding, kernel_size = kernel_size, bias = bias)
        self.B = mi.torch.nn.BatchNorm2d(out_channels)
        self.Conv.to(mi.my_device_0)
        self.B.to(mi.my_device_0)
    
    def forward(self, X) :
        X = self.Conv(X)
        return(self.B(X))

skip_connections_RN18 = {'in': [2, 4, 7, 9, 12, 14, 17, 19], 'out': [3, 5, 8, 10, 13, 15, 18, 20]}


# if not precised then the function of the skip connection is the Identity
skip_fct_RN18 = {}


##############################################
############# LAYERS #########################

reduction = G.reduction
a, b, c, d = [64, 128, 256, 512]
a0, b0, c0, d0 = [int(a/reduction), int(b/reduction), int(c/reduction), int(d/reduction)]
#a0, b0, c0, d0
depth_seuil = {2 : 3, 4 : 3, 7 : 6, 9 : 6,
                   12 : 12, 14 : 12, 17 : 24, 19 : 24}

nbr_pass = G.nbr_pass
ainf, binf, cinf, dinf = [int(a0 + nbr_pass*depth_seuil[2]), int(b0 + nbr_pass*depth_seuil[7]), int(c0 + nbr_pass*depth_seuil[12]), int(d0 + nbr_pass*depth_seuil[17])]


layer_name_RN18 = {1 : 'CB',  2 : 'CB', 3 : 'CB', 4 : 'CB', 5 :'CB',
                  6 : 'CB',  7 : 'CB', 8 : 'CB', 9 : 'CB',
                  10 : 'CB', 11 : 'CB', 12 : 'CB', 13 : 'CB',
                  14 : 'CB', 15 : 'CB', 16 : 'CB', 17 : 'CB',
                  18 : 'CB', 19 : 'CB', 20 : 'CB',
                  21 : 'L'}


skeleton_RN18 = {0: {}, 
1: {'in_channels': 3, 'out_channels': a, 'kernel_size': (3, 3), 'padding': 1, 'stride': 1, 'bias': False}, 
 
2: {'in_channels': a, 'out_channels': a0, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': True}, 
3: {'in_channels': a0, 'out_channels': a, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': True}, 
4: {'in_channels': a, 'out_channels': a0, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': True}, 
5: {'in_channels': a0, 'out_channels': a, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': True}, 
6 : {'in_channels' : a, 'out_channels' : b, 'kernel_size' : (1, 1), 'padding' : (0, 0), 'stride' : (1, 1)},
                    
7: {'in_channels': b, 'out_channels': b0, 'kernel_size': (3, 3),'padding': (1, 1), 'stride': (1, 1), 'bias': True}, 
8: {'in_channels': b0, 'out_channels': b, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': True}, 
9: {'in_channels': b, 'out_channels': b0, 'kernel_size': (3, 3),'padding': (1, 1), 'stride': (1, 1), 'bias': True}, 
10:{'in_channels': b0, 'out_channels': b, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': True}, 
11:{'in_channels' : b, 'out_channels' : c, 'kernel_size' : (1, 1), 'padding' : (0, 0), 'stride' : (1, 1)},
                       
12: {'in_channels': c, 'out_channels': c0, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': True}, 
13: {'in_channels': c0, 'out_channels': c, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': True}, 
14: {'in_channels': c, 'out_channels': c0, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': True}, 
15: {'in_channels': c0, 'out_channels': c, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': True}, 
16:{'in_channels' : c, 'out_channels' : d, 'kernel_size' : (1, 1), 'padding' : (0, 0), 'stride' : (1, 1)},

17: {'in_channels': d, 'out_channels': d0, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': True}, 
18: {'in_channels': d0, 'out_channels': d, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': True}, 
19: {'in_channels': d, 'out_channels': d0, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': True}, 
20: {'in_channels': d0, 'out_channels': d, 'kernel_size': (3, 3), 'padding': (1, 1), 'stride': (1, 1), 'bias': True}, 
21: {'size' : 100}}



##############################################
####### ACTIVATION FUNCTIONS #################


def fct_17(X) :
    return(mi.torch.nn.AdaptiveAvgPool2d(output_size = (1, 1))(mi.torch.nn.ReLU()(X)))

#change_channels_depth = [6, 11, 16]
change_channels_depth = [6, 11, 16]

fct_RN18 = {depth : mi.torch.nn.ReLU() for depth in range(1, 21) if not(depth in change_channels_depth)}
fct_RN18.update({depth : mi.torch.nn.AvgPool2d(kernel_size = (2, 2), stride = (2, 2), padding  = (0, 0)) for depth in change_channels_depth})
fct_RN18.update({20 : mi.torch.nn.AdaptiveAvgPool2d(output_size = (1, 1))})
fct_RN18.update({21 : mi.torch.nn.Identity()})

##############################################

#T_j_depth = [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19]