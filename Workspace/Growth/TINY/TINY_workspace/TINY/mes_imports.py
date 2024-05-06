import os, sys, time, pickle, gc, math
from typing import List, Optional, Dict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy, torch
from scipy.sparse.linalg import svds
from scipy import stats
from torch.optim import _functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch import Tensor

from collections import defaultdict
from tqdm.notebook import tqdm
# from tabulate import tabulate
import copy

import UTILS
#from statsmodels.multivariate.multivariate_ols import multivariate_stats
#import VGG_glissant_uncouple_lr_gaussien_GM_RS_RP_amplitude_racine as VGG_uncouple






XTX_depth1 = {'NG' : {'XTX' : None, 'MSM' : None, 'batch_size': 0, 'indices' : []},
              'Add' : {'XTX' : None, 'MSM' : None, 'batch_size' : 0, 'indices' : []}
              }

if torch.cuda.is_available():
    device = torch.device('cuda')
    my_device = 'cuda:0'
    
    my_device_0 = 'cuda:0'
    if torch.cuda.device_count() > 1:
        my_device_1 = 'cuda:1'
    else:
        my_device_1 = 'cuda:0'
else:
    device = torch.device('cpu')
    my_device = 'cpu'
    my_device_0 = 'cpu'
    my_device_1 = 'cpu'
