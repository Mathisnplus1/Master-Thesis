- gitignore
- fix imports
- documentation

Questions:
- pk ne pas update les paramètres avec dW* ? -> car une fois les neurones ajouté dW* n'est plus valide (according to stella this is not necessary true)
- est ce que dW* et dL/dW sont très différents ?
- est ce que les données pour estimer dW* et (N, S) doivent être les mêmes ?
- pk rescale_alpha_omega ?

Details:
- TINY:
    - lign 197: get batch problem when we enter first condition
    - same problem at lign 356
    - forward:
        - (V) lign 270: why where is a class attribute here? -> probably to keep track of the process
        - lign 301: what append if we have skip connections in linera layers ? -> to FIX
    - complexity_model:
        - hard coded image size to fix
    - train batch unused limite_temps, pas de validation utilisée

SOLVE_EB:
    - SVD_Smoins1demiN : rename, pk ne pas prendre directement S^{-1/2} en entrée ?
    - S_1demiN : rename, qu'est ce que QVT ? selected index ?
    - __MSM_MDV : second bloc de condition elif illogic



TINY parameters:
- init_X_shape -> input_shape: tuple
    shape of the input tensor
- deep -> depth: int
    number of layers
- layer_name -> layer_type: dict[int, str]
    type of each layer ({n: type}) where n is the layer number and type is the layer type with the following options:
        - 'LD' Linear with dropout
        - 'L' Linear
        - 'CB' Convolutional with batch normalization
- skip_connections: dict[str, list[int]]
    skip connections between layers where skip_connections['out'][i] is the output layer and skip_connections['in'][i] is the input layer of the i-th skip connection
- skip_fct: dict[int, torch.nn.functional | torch.nn.Module]
    function to apply to the skip connection
- fct: dict[int, torch.nn.functional | torch.nn.Module]
    function to apply to the output of each layer


Papier coquilles:
- page 13, A.2 ajouter derivé sous l'intégrale ?
- page 13, reférence valide ?
- page 4, equation 1: $\not \frac{\partial f_{\theta}}{\partial \theta}$
- page 4, figure 2: one $\mathcal{T_A}$ en trop
- page 5, proposition 3: manque couleur sur V goal
- page 18, rappeler la def de dW*
- page 19, R -> K
