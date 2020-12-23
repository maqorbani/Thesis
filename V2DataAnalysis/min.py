import numpy as np
import os

min = 100
max = 0

for i in os.listdir():
    if os.path.exists(f'{i}/{i}.gz'):
        file = np.loadtxt(f'{i}/{i}.gz')
        min = file.min() if file.min() < min else min
        max = file.max() if file.max() > max else max

        print(min, ', ', max)

with open('min-max.txt', 'w') as f:
    f.write(str(min) + ', ' + str(max))
