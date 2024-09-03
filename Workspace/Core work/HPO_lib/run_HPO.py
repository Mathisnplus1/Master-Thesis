from HPOs.greedy_HPO_exp import call_greedy_HPO, call_greedy_HPO_for_EWC
from HPOs.cheated_HPO import call_cheated_HPO


def run_HPO(HPO_settings, method_settings, benchmark_settings, benchmark, device, global_seed) :
    if HPO_settings["HPO_name"] == "greedy_HPO" :
        if method_settings["method_name"] == "EWC" :
            return call_greedy_HPO_for_EWC(HPO_settings, method_settings, benchmark_settings, benchmark, device, global_seed)
        else :
            return call_greedy_HPO(HPO_settings, method_settings, benchmark_settings, benchmark, device, global_seed)
    if HPO_settings["HPO_name"] == "cheated_HPO" :
        return call_cheated_HPO(HPO_settings, method_settings, benchmark_settings, benchmark, device, global_seed)