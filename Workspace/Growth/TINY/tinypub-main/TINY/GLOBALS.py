activity = {}
dico_mask_tensor_t = {}
dico_tt = {}
dico_F = {}
dico_T_permute_flat = {}
dico_T_flat_sparse = {}
outputs = {}

batch_size = 32
lr = 1e-2


def print_globals():
    dictionaries = {
        "activity": activity,
        "dico_T": dico_mask_tensor_t,
        "dico_S": dico_tt,
        "dico_F": dico_F,
        "dico_T_permute_flat": dico_T_permute_flat,
        "dico_T_flat_sparse": dico_T_flat_sparse,
        "outputs": outputs
    }

    for k, v in dictionaries.items():
        if v:
            print('=' * 20)
            print(f"Global variable: {k}")
            print('keys:', v.keys())
            assert isinstance(v, dict), f"Expected a dictionary, got {type(v)}"
            try:
                print('shapes:', {key: val.shape for key, val in v.items()})
            except AttributeError:
                pass

    print('\n', '-' * 60, '\n')


'''
paths = ['resultats/1quart_tbadd_1/1', 'resultats/1quart_tbadd_1/2']
architecture_growth = 'Our'
nbr_epochs_betw_adding = 1
reduction = 4
rescale = 'DE'
nbr_pass = 8
lambda_method = 0
'''

'''
paths = ['resultats/1quart_tbadd_1/3', 'resultats/1quart_tbadd_1/4']

architecture_growth = 'GradMax'
nbr_epochs_betw_adding = 1
reduction = 4
rescale = 'theta'
nbr_pass = 8
lambda_method = 1e-3
'''


'''
paths = ['resultats/1quart_tbadd_0.25/1', 'resultats/1quart_tbadd_0.25/2']
architecture_growth = 'Our'
nbr_epochs_betw_adding = 0.25
reduction = 4
rescale = 'DE'
nbr_pass = 8
lambda_method = 0
'''

'''
paths = ['resultats/1quart_tbadd_0.25/3', 'resultats/1quart_tbadd_0.25/4']

architecture_growth = 'GradMax'
nbr_epochs_betw_adding = 0.25
reduction = 4
rescale = 'theta'
nbr_pass = 8
lambda_method = 1e-3
'''

'''
paths = ['resultats/1soixantequatre_tbadd_0.25/1', 'resultats/1soixantequatre_tbadd_0.25/2']
architecture_growth = 'Our'
nbr_epochs_betw_adding = 0.25
reduction = 64
rescale = 'DE'
nbr_pass = 21
lambda_method = 0
'''
'''
paths = ['resultats/1soixantequatre_tbadd_0.25/3', 'resultats/1soixantequatre_tbadd_0.25/4']

architecture_growth = 'GradMax'
nbr_epochs_betw_adding = 0.25
reduction = 64
rescale = 'theta'
nbr_pass = 21
lambda_method = 1e-3
'''

'''

paths = ['resultats/1soixantequatre_tbadd_1/1', 'resultats/1soixantequatre_tbadd_1/2']
architecture_growth = 'Our'
nbr_epochs_betw_adding = 1
reduction = 64
rescale = 'DE'
nbr_pass = 21
lambda_method = 0

'''


paths = ['resultats/1soixantequatre_tbadd_1/3', 'resultats/1soixantequatre_tbadd_1/4']
architecture_growth = 'GradMax'
nbr_epochs_betw_adding = 1
reduction = 64
rescale = 'theta'
nbr_pass = 21
lambda_method = 1e-3

