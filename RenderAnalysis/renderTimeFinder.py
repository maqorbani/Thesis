# %%
import re
import os
import numpy as np

pattern = re.compile(r'(\d.\d{3})r')

times = []

for i in os.listdir():
    if os.path.isdir(i):
        with open(f'{i}/error.log') as f:
            rTime = f.readlines()[-1]

    print(float(re.findall(pattern, rTime)[0]))

    times.append(float(re.findall(pattern, rTime)[0]))

times = np.array(times)
