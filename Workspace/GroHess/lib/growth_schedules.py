

def get_CL_schedule(num_neurons):
    growth_schedule = [[0, 9, "fc1", num_neurons],[0, 18, "fc2", num_neurons],
                       [0, 10, "fc1", num_neurons],[0, 20, "fc2", num_neurons],
                       [0, 11, "fc1", num_neurons],[0, 22, "fc2", num_neurons],
                       [0, 10, "fc1", num_neurons],[0, 20, "fc2", num_neurons],
                       [0, 10, "fc1", num_neurons],[0, 20, "fc2", num_neurons]]
    return growth_schedule