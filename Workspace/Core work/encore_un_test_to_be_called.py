import sys
import time
import torch
import json

input_data = json.loads(sys.stdin.read())


# Parameters specific to the method
method_settings = {"method_name" : "GroHess",
                   "grow_from" : "output",
                   "hessian_percentile" : 95,
                   "grad_percentile" : 95,
                   "num_inputs" : 28*28,
                   "num_hidden_root" : 1000,
                   "num_outputs" : 10,
                   "loss_name" : "CE",
                   "optimizer_name" : "Adam"}

sys.path.append("HPO_lib")
sys.path.append("Methods/" + method_settings["method_name"])

from HPO_lib.abstract_torch import get_device
from Methods.GroHess.lib.method import initialize_model

def print_gpu_memory(device):
    print(f"Allocated memory: {torch.cuda.memory_allocated(device) / (1024 ** 2)} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved(device) / (1024 ** 2)} MB")

device = get_device(0)

time.sleep(2)
dummy_input = torch.randn(1, 28*28).to(device)
print("After initializing dummy_input:")
print_gpu_memory(device)

time.sleep(2)
model1 = initialize_model(method_settings, 88).to(device)
print("After initializing model1:")
print_gpu_memory(device)

time.sleep(2)
model2 = initialize_model(method_settings, 89).to(device)
print("After initializing model2:")
print_gpu_memory(device)

time.sleep(2)
model2(dummy_input)
print("After forwarding model2:")
print_gpu_memory(device)

time.sleep(2)
model3 = initialize_model(method_settings, 90).to(device)
print("After initializing model3:")
print_gpu_memory(device)


print(input_data[2])
