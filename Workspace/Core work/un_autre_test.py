import time

try :
    with open('test.txt', 'r') as f:
        value = print(f.read())
except :
    value = 0

time.sleep(5)

if value <= 5 :
    with open('test.txt', 'w') as f:
        f.write(value+1)
    