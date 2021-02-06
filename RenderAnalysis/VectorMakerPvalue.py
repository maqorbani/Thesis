'''
DO NOT RUN THIS ALL AT ONCE!
'''
# %%
import os
import io
import numpy as np
import concurrent.futures

# %%
os.chdir('V2SouthAB0/')
# %%
nCPU = 32
HDRs = os.listdir()
divisionCPU = len(HDRs) // nCPU
HDRsDict = {}

minMaxDict = {i: [100, 0] for i in range(nCPU)}

for i in range(nCPU):
    HDRsDict['HDR'+str(i)] = HDRs[i*divisionCPU:divisionCPU*(i+1)]

for i, HDR in enumerate(HDRs[nCPU * divisionCPU:]):
    HDRsDict['HDR'+str(i)].append(HDR)

badDirs = []


# %%
def tensorExtractor(hdrs, index):
    for hdr in hdrs:
        # print(os.getcwd())
        if os.system(
                f'pvalue -h -d -H -b -o {hdr}/{hdr}_1.HDR > {hdr}/{hdr}.txt'):
            print(hdr, ' is bad image.')
            badDirs.append(hdr)
            os.remove(f'{hdr}/{hdr}.txt')
            continue
        else:
            print(hdr)
        with io.open(f'{hdr}/{hdr}.txt', 'r', encoding=None) as f:
            vector = np.loadtxt(f, delimiter='\n')
            MIN, MAX = vector.min(), vector.max()
            if MIN < minMaxDict[index][0]:
                minMaxDict[index][0] = MIN
            if MAX > minMaxDict[index][1]:
                minMaxDict[index][1] = MAX
            np.savetxt(f'{hdr}/{hdr}.gz', vector)
        os.remove(f'{hdr}/{hdr}.txt')


# tensorExtractor(os.listdir())
with concurrent.futures.ThreadPoolExecutor() as exec:
    res = [exec.submit(tensorExtractor, HDRsDict['HDR'+str(i)], i)
           for i in range(nCPU)]

# %%
# minMax to numpy here
minMax = np.array(list(minMaxDict.values()))
MIN = minMax[:, 0].min()
MAX = minMax[:, 1].max()

with open('min-max.txt', 'w') as f:
    f.write(f'{MIN}, {MAX}')

# %%
# os.chdir('../')
# %%
with open('badDirs.txt', 'w') as f:
    # [f.write(i + ',') for i in badDirs]
    f.write(', '.join(badDirs))

'''
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
'''
