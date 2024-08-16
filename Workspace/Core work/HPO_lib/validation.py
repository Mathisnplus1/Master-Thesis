try :
    from validations.greedy_validation import greedy_validate
except :
    from validations.greedy_validation import greedy_validate
from validations.cheated_validation import cheated_validate


def validate(HPO_settings, benchmarks_list, benchmark_settings, method_settings, best_params_list, device, global_seed) :
    if HPO_settings["HPO_name"] == "greedy_HPO" :
        return greedy_validate(benchmarks_list, benchmark_settings, method_settings, best_params_list, device, global_seed)
    if HPO_settings["HPO_name"] == "cheated_HPO" :
        return cheated_validate(benchmarks_list, benchmark_settings, method_settings, best_params_list, device, global_seed)