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

def is_pytorch_object(obj):
    pytorch_classes = (
        torch.Tensor,
        torch.nn.Module,
        torch.optim.Optimizer,
        # Add more PyTorch classes as needed
    )
    return isinstance(obj, pytorch_classes)

def la_func(unint, unstring, model) :
    global dic
    signature = inspect.signature(la_func)
    names = [param.name for param in signature.parameters.values()]
    values = locals()
    dic = {name: values[name] for name in names}
    for key, value in dic.items() :
        if is_pytorch_object(value) :
            torch.save(value, f'logs/{key}.pt')
        else :
            with open(f'logs/{key}.pkl', 'wb') as f:
                pickle.dump(value, f)
    result = subprocess.run(["python", "encore_un_test_to_be_called.py"], 
               input=json.dumps(names).encode(),
               capture_output=True,
               check=True)
    return result

result = la_func(1, "c'est pi ça", torch.nn.Linear(10, 10))

time.sleep(1)

print("###########################")
print(result.stdout.splitlines())
print("###########################")

# remove the files we created
for key, value in dic.items() :
    if is_pytorch_object(value) :
        subprocess.run(["rm", f'logs/{key}.pt'])
    else :
        subprocess.run(["rm", f'logs/{key}.pkl'])
   
#print("DONE WITH SCRIPT:")
#print_gpu_memory(device)

#time.sleep(1)
