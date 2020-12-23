# This file is not completed
import numpy as np
import os
import concurrent.futures

nCPU = 1
folder = 'ab0'
HDRs = os.listdir(f'{folder}/')

divisionCPU = len(HDRs) // nCPU
HDRdict = {}
for i in range(nCPU):
    HDRdict['sky'+str(i)] = HDRs[i*divisionCPU:divisionCPU*(i+1)]

for i, sky in enumerate(HDRs[nCPU * divisionCPU:]):
    HDRdict['sky'+str(i)].append(sky)
print(HDRdict)


def main(dirs):
    pass


with concurrent.futures.ProcessPoolExecutor() as executor:
    results = [executor.submit(main, list(
        HDRdict['sky'+str(i)]), i) for i in range(nCPU)]

    for f in concurrent.futures.as_completed(results):
        print(f.result())
