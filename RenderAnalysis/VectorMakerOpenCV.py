# %%
import os
import cv2
import concurrent.futures
import numpy as np

nCPU = 10  # Number of CPUs to split work to

os.chdir('EastV3')
print('Current working dir is: ' + os.getcwd())

HDRs = []
for i in os.listdir():
    if os.path.isdir(i):
        HDRs.append(i)
    else:
        # os.remove(i)
        print(i)

divisionCPU = len(HDRs) // nCPU

hdrDict = {}
for i in range(nCPU):
    hdrDict['HDR'+str(i)] = HDRs[i*divisionCPU:divisionCPU*(i+1)]

for i, HDR in enumerate(HDRs[nCPU * divisionCPU:]):
    hdrDict['HDR'+str(i)].append(HDR)
# print(hdrDict)

badImage = []


def tensorFromHDR(HDRs, j):
    for i in HDRs:
        try:
            cv2.imread(i + f'/{i}_1.HDR', -1).sum(axis=-1)
        except cv2.error as e:
            print(e)
            badImage.append(i)
            continue
        image = cv2.imread(i + f'/{i}_1.HDR', -1).sum(axis=-1)
        np.savetxt(i + f'/{i}.gz', image)
    return 'Process #' + str(j) + ' is done!'


# tensorFromHDR(HDRs, 0)
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = [executor.submit(tensorFromHDR, list(
        hdrDict['HDR'+str(i)]), i) for i in range(nCPU)]

    for f in concurrent.futures.as_completed(results):
        print(f.result())

print(len(badImage))
with open('badImages.txt', 'w') as f:
    f.write(badImage)
