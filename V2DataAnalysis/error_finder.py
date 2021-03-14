# %%
import os
import numpy as np
import concurrent.futures

nCPU = 32

key = list(np.loadtxt('data/key.txt', dtype='str'))  # HOY for each key

HDRs = os.listdir('ab4')
HDRs.remove('min-max.txt')

HDRsDict = {}
divisionCPU = len(HDRs) // nCPU

for i in range(nCPU):
    HDRsDict['HDR'+str(i)] = HDRs[i*divisionCPU:divisionCPU*(i+1)]

for i, HDR in enumerate(HDRs[nCPU * divisionCPU:]):
    HDRsDict['HDR'+str(i)].append(HDR)

Ev = np.zeros((len(HDRs), 2))
dgp = np.zeros((len(HDRs), 2))


# %%
def dgp_Ev_extractor(hdrs):
    for i in hdrs:
        index = key.index(i)

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

        print(i)


with concurrent.futures.ThreadPoolExecutor() as exec:
    res = [exec.submit(dgp_Ev_extractor, HDRsDict['HDR'+str(i)])
           for i in range(nCPU)]

# %%
