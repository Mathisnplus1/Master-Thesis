import subprocess
import json


global_seeds = [89,90,91,92]

for global_seed in global_seeds :
    subprocess.run(["python", "script_greedy_HPO_old_thresholding.py"], 
                input=json.dumps(global_seed).encode(),
                capture_output=False,
                check=False)