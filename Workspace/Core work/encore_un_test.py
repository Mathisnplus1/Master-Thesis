import sys
import time
import subprocess
import torch
import json
import inspect
import pickle

sys.path.append("HPO_lib")

from HPO_lib.abstract_torch import get_device

device = get_device(0)

def print_gpu_memory(device):
    print(f"Allocated memory: {torch.cuda.memory_allocated(device) / (1024 ** 2)} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved(device) / (1024 ** 2)} MB")

def la_func(unint, unstring, model) :
    signature = inspect.signature(la_func)
    names = [param.name for param in signature.parameters.values()]
    values = locals()
    dic = {name: values[name] for name in names}
    for key, value in dic.items() :
        with open(f'logs/{key}.pkl', 'wb') as f:
            pickle.dump(value, f)
    result = subprocess.run(["python", "encore_un_test_to_be_called.py"], 
               capture_output=True,
               check=True)
    return result

result = la_func(1, "c'est pi ça", torch.nn.Linear(10, 10))


time.sleep(1)

print("###########################")
print(result.stdout.splitlines()[-1])
print("###########################")

print("DONE WITH SCRIPT:")
print_gpu_memory(device)

time.sleep(1)
