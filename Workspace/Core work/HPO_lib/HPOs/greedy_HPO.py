def call_greedy_HPO(n_trials) :
    # Initialize model
    model = ANN(num_inputs, num_hidden_root, num_outputs, random_seed).to(device)

    # Intialize mask
    if grow_from == "input" :
        overall_masks = [np.ones_like(model.fc1.weight.data.cpu().numpy()),
                        np.ones_like(model.fc2.weight.data.cpu().numpy())]
    else :
        overall_masks = [np.ones_like(model.fc2.weight.data.cpu().numpy()),
                        np.ones_like(model.fc3.weight.data.cpu().numpy())]
            
    # Initialize variable to store the best HPs and the scores
    best_params_list = []
    test_accs_matrix = np.zeros((num_tasks, num_tasks))

    for task_number in range(0,num_tasks) :

        # Verbose
        print("\n" + "-"*50)
        print(f"LEARNING TASK {task_number+1}")

        # Perform HPO
        storage = optuna.storages.InMemoryStorage()
        study = optuna.create_study(storage=storage,
                                    study_name=f"Search number {task_number+1}",
                                    sampler=optuna.samplers.TPESampler(seed=random_seed),
                                    direction = "maximize")
        
        is_first_task = True if task_number==0 else False
        params = overall_masks, is_first_task
        partial_objective = partial(objective, model, task_number, params, device)
        study.optimize(partial_objective,
                    n_trials=n_trials,
                    timeout=3600)

        # Retrain and save a model with the best params
        best_params = study.best_trial.params
        best_params_list += [best_params]
        overall_masks = retrain_and_save_with_best_HPs(model, params, best_params, train_loaders_list[task_number]) 
        
        # Test on each task
        for j in range(num_tasks) :
            test_accs_matrix[task_number,j] = round(test(model, test_loaders_list[j], batch_size, device),2)
    
    return test_accs_matrix, best_params_list, model