import numpy as np
import random
a = np.array([1,2,3,4,5,6])
b = a

for i in range(3):
    ran = random.randint(10,15)
    b[0:-1] = b[1:].copy()
    b[-1] = ran
    print(b)