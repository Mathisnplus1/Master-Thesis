Remarques sur le contenu du repo accessible à https://gitlab.inria.fr/mverbock/tinypub à la date du 3 mai 2024 et sur l'article (version que vous m'avez envoyée par mail)

---

## Repo

1. TINY/TINY.py : 
    - le nom des variables mélange anglais et français
    - nom des variables et commentaires pas toujours intuitifs
2. DEMO/MLP/MLP on MNIST.ipynb :
    - Le notebook est globalement difficile à lire, avec des noms de variables mélangeant anglais et français et plutôt peu explicites : ```A_te```, ```best_depth``` ou ```nbr_parameters_avant```
    - Le plot de la cellule 12 indique "Accuracy test" en ordonée mais la train loss et la test loss sont affichées
    - Cellule 13 : ```name_file_expe``` prend la valeur "resultats/" ce qui empêche la cellule 16 de s'exécuter (à la ligne 35, ```df_tracker```, ne peut pas être enregistré parce que le répertoire "resultats/" n'existe pas)
    - Une fois que le répertoire "resultats/" est créé, le notebook s'exécute sans encombre

## Article

1. Section 2.3 : Given an sample -> Given a sample
2. Section 3.2 : 
    - Le mot "redundancy" apparait 6 fois dans l'introdcution, et il semble qu'éviter la redondance des neurones ajoutés avec ceux existants soient une part importante de la contribution du papier. Il ne réapparait qu'une seule fois, dans le papier, lorsque il est mentionné que la méthode GradMax cherche, elle aussi, à minimiser la loss 7 mais sans considérer la redondance. Mais entre temps, la redondance ne fait l'objet d'une définition formelle, et je ne comprend pas clairement comment elle est prise en compte.
    - La référence à l'appendice, à la dernière ligne, ne marche pas.
2. Section 4 : 
    - Je ne comprends pas comment les propriétés 4.1 à 4.3 assurent que l'approche greedy (consistant à ajouter les meilleurs neurones à chaque étape de croissance indépendamment de ceux qu'on ajoutera par la suite) ne sera pas bloquée à un état sous-optimal. Le fait qu'à l'étape $t$ de la croissance du réseau, il existe toujours des neurones à ajouter diminuant strictement la loss de $l_{t-1}$ à $l_t$ n'implique pas que choisir ces neurones à chaque étape $t$ conduira la séquence $l_1,l_2,...,l_t$ à converger vers $\min_{f}(\mathcal{L}(f))$. Ou peut être que je ne comprends pas les phrases débutant la section 4 :

        *One might wonder whether a greedy approach on layer growth might get stuck in a non-optimal state. We derive the following series of propositions in this regard.*

    - Est-ce qu'une approche inspirée de la programmation dynamique a été envisagée ? Comme on part d'une petite architecture, le coup associé à l'exploration de plusieurs croissances en parallèle ne serait peut-être pas si dément..?
