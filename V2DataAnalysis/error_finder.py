# %%
import os
import numpy as np

HDRs = os.listdir('ab4')
HDRs.remove('min-max.txt')

Ev = np.zeros((len(HDRs), 2))
dgp = np.zeros((len(HDRs), 2))

for index, i in enumerate(HDRs[:4]):
    os.system(f'evalglare -V -vtv -vh 110 -vv 77.57\
        E:/Rendered_bak/V2SouthAB4/{i}/{i}_1.HDR >\
            E:/Rendered_bak/V2SouthAB4/{i}/{i}_g.txt')
    os.system(f'evalglare -vtv -vh 110 -vv 77.57\
        E:/Rendered_bak/V2SouthAB4/{i}/{i}_1.HDR >>\
            E:/Rendered_bak/V2SouthAB4/{i}/{i}_g.txt')

    with open(f'E:/Rendered_bak/V2SouthAB4/{i}/{i}_g.txt', 'r') as f:
        Ev[index, 0] = float(f.readline().strip())
        data = f.readline()
        dgp[index, 0] = float(data.split(': ')[-1].strip().split(' ')[0])

    os.system(f'evalglare -V -vtv -vh 110 -vv 77.57\
        ab4/{i}/{i}_2.HDR > ab4/{i}/{i}_g.txt')
    os.system(f'evalglare -vtv -vh 110 -vv 77.57\
        ab4/{i}/{i}_2.HDR >> ab4/{i}/{i}_g.txt')

    with open(f'ab4/{i}/{i}_g.txt', 'r') as f:
        Ev[index, 1] = float(f.readline().strip())
        data = f.readline()
        dgp[index, 1] = float(data.split(': ')[-1].strip().split(' ')[0])
