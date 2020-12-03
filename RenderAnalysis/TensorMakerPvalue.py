'''
DO NOT RUN THIS ALL AT ONCE!
'''
# %%
import os
import io
import numpy as np
import concurrent.futures

os.chdir('EastV3/')
nCPU = 20
HDRs = os.listdir()
divisionCPU = len(HDRs) // nCPU
HDRsDict = {}

for i in range(nCPU):
    HDRsDict['HDR'+str(i)] = HDRs[i*divisionCPU:divisionCPU*(i+1)]

for i, HDR in enumerate(HDRs[nCPU * divisionCPU:]):
    HDRsDict['HDR'+str(i)].append(HDR)

badDirs = []

# %%
def tensorExtractor(hdrs):
    for hdr in hdrs:
        print(os.getcwd())
        if os.system(f'pvalue -h -d -H -b {hdr}/{hdr}_1.HDR > {hdr}/{hdr}.txt'):
            print(hdr, ' is bad image.')
            badDirs.append(hdr)
            os.remove(f'{hdr}/{hdr}.txt')
            continue
        else:
            print(hdr)
        with io.open(f'{hdr}/{hdr}.txt', 'r', encoding=None) as f:
            np.savetxt(f'{hdr}/{hdr}.gz', np.loadtxt(f, delimiter='\n'))
        os.remove(f'{hdr}/{hdr}.txt')


# tensorExtractor(os.listdir())
with concurrent.futures.ThreadPoolExecutor() as exec:
    res = [exec.submit(tensorExtractor, HDRsDict['HDR'+str(i)]) for i in range(nCPU)]

# %%
os.chdir('../')
# %%
with open('badDirs.txt', 'w') as f:
    # [f.write(i + ',') for i in badDirs]
    f.write(', '.join(badDirs))

# %%
with open('skyResearch/key.txt', 'r') as f:
    keys = f.read()

keys = keys.split('\n')

goodie = []
for bad in badDirs:
    if bad in keys:
        goodie.append(bad)

with open('skiesToCreate.txt', 'w') as f:
    [f.write(i + ',') for i in goodie]
