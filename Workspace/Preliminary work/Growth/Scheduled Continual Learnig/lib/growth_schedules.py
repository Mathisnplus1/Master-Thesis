

def get_handcrafted_schedule(num_neurons):
    growth_schedule_1 = [["fc2", num_neurons],["fc2", num_neurons],["fc2", num_neurons],
                        ["fc2", num_neurons],["fc2", num_neurons],["fc2", num_neurons],
                        ["fc2", num_neurons],["fc2", num_neurons],["fc2", num_neurons],
                        ["fc2", num_neurons],["fc1", num_neurons],["fc1", num_neurons],
                        ["fc1", num_neurons],["fc1", num_neurons],["fc1", num_neurons],
                        ["fc1", num_neurons],["fc1", num_neurons],["fc1", num_neurons],
                        ["fc1", num_neurons],["fc1", num_neurons]]
    growth_schedule_2 = [["fc2", num_neurons],["fc2", num_neurons],["fc2", num_neurons],
                        ["fc2", num_neurons],["fc1", num_neurons],["fc2", num_neurons],
                        ["fc2", num_neurons],["fc2", num_neurons],["fc2", num_neurons],
                        ["fc1", num_neurons],["fc2", num_neurons],["fc1", num_neurons],
                        ["fc1", num_neurons],["fc1", num_neurons],["fc1", num_neurons],
                        ["fc2", num_neurons],["fc1", num_neurons],["fc1", num_neurons],
                        ["fc1", num_neurons],["fc1", num_neurons]]
    growth_schedule_3 = [["fc2", num_neurons],["fc2", num_neurons],["fc2", num_neurons],
                        ["fc1", num_neurons],["fc1", num_neurons],["fc2", num_neurons],
                        ["fc2", num_neurons],["fc2", num_neurons],["fc1", num_neurons],
                        ["fc1", num_neurons],["fc2", num_neurons],["fc2", num_neurons],
                        ["fc1", num_neurons],["fc1", num_neurons],["fc1", num_neurons],
                        ["fc2", num_neurons],["fc2", num_neurons],["fc1", num_neurons],
                        ["fc1", num_neurons],["fc1", num_neurons]]
    growth_schedule_4 = [["fc2", num_neurons],["fc1", num_neurons],["fc2", num_neurons],
                        ["fc1", num_neurons],["fc2", num_neurons],["fc1", num_neurons],
                        ["fc2", num_neurons],["fc1", num_neurons],["fc2", num_neurons],
                        ["fc1", num_neurons],["fc2", num_neurons],["fc1", num_neurons],
                        ["fc2", num_neurons],["fc1", num_neurons],["fc2", num_neurons],
                        ["fc1", num_neurons],["fc2", num_neurons],["fc1", num_neurons],
                        ["fc2", num_neurons],["fc1", num_neurons]]
    growth_schedule_5 = [["fc1", num_neurons],["fc2", num_neurons],["fc1", num_neurons],
                        ["fc2", num_neurons],["fc1", num_neurons],["fc2", num_neurons],
                        ["fc1", num_neurons],["fc2", num_neurons],["fc1", num_neurons],
                        ["fc2", num_neurons],["fc1", num_neurons],["fc2", num_neurons],
                        ["fc1", num_neurons],["fc2", num_neurons],["fc1", num_neurons],
                        ["fc2", num_neurons],["fc1", num_neurons],["fc2", num_neurons],
                        ["fc1", num_neurons],["fc2", num_neurons]]
    growth_schedule_6 = [["fc1", num_neurons],["fc1", num_neurons],["fc1", num_neurons],
                        ["fc2", num_neurons],["fc2", num_neurons],["fc1", num_neurons],
                        ["fc1", num_neurons],["fc1", num_neurons],["fc2", num_neurons],
                        ["fc2", num_neurons],["fc1", num_neurons],["fc1", num_neurons],
                        ["fc2", num_neurons],["fc2", num_neurons],["fc2", num_neurons],
                        ["fc1", num_neurons],["fc1", num_neurons],["fc2", num_neurons],
                        ["fc2", num_neurons],["fc2", num_neurons]]
    growth_schedule_7 = [["fc1", num_neurons],["fc1", num_neurons],["fc1", num_neurons],
                        ["fc1", num_neurons],["fc2", num_neurons],["fc1", num_neurons],
                        ["fc1", num_neurons],["fc1", num_neurons],["fc1", num_neurons],
                        ["fc2", num_neurons],["fc1", num_neurons],["fc2", num_neurons],
                        ["fc2", num_neurons],["fc2", num_neurons],["fc2", num_neurons],
                        ["fc1", num_neurons],["fc2", num_neurons],["fc2", num_neurons],
                        ["fc2", num_neurons],["fc2", num_neurons]]
    growth_schedule_8 = [["fc1", num_neurons],["fc1", num_neurons],["fc1", num_neurons],
                        ["fc1", num_neurons],["fc1", num_neurons],["fc1", num_neurons],
                        ["fc1", num_neurons],["fc1", num_neurons],["fc1", num_neurons],
                        ["fc1", num_neurons],["fc2", num_neurons],["fc2", num_neurons],
                        ["fc2", num_neurons],["fc2", num_neurons],["fc2", num_neurons],
                        ["fc2", num_neurons],["fc2", num_neurons],["fc2", num_neurons],
                        ["fc2", num_neurons],["fc2", num_neurons]]
    growth_schedules = [growth_schedule_1, growth_schedule_2, growth_schedule_3, growth_schedule_4,
                        growth_schedule_5, growth_schedule_6, growth_schedule_7, growth_schedule_8]
    
    return growth_schedules


def get_CL_schedule(num_neurons):
    growth_schedule = [[[0, 9, "fc1", num_neurons],[0, 18, "fc2", num_neurons]],
                       [[0, 10, "fc1", num_neurons],[0, 20, "fc2", num_neurons]],
                       [[0, 11, "fc1", num_neurons],[0, 22, "fc2", num_neurons]],
                       [[0, 10, "fc1", num_neurons],[0, 20, "fc2", num_neurons]],
                       [[0, 10, "fc1", num_neurons],[0, 20, "fc2", num_neurons]]]
    return [growth_schedule]