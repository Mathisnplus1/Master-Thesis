import pickle

with open("logs/task_number.pkl", "rb") as f :
    task_number = pickle.load(f)
print(task_number)