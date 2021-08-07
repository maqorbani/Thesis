# %%
import os
import subprocess
import shutil
import concurrent.futures
import numpy as np
import io
import time

# %%
# Variables
ab = 4  # Either 4 or 0
nCPU = 64
nCores = 1  # Number of rtrace threads
imgDims = 256  # output would be: imgDims * imgDims/2
# End of Variables

# os.chdir('Desktop/TheRender/')
print('Current working dir is: ' + os.getcwd())

# Set the skies
skies = [i[:-4] for i in os.listdir('../skies/')]
# skies = skies[:3]  # To test a few number

# For min-max finder
minMaxDict = {i: [100, 0] for i in range(nCPU)}

# Overview
print(f'Number of HDRs to be rendered: {len(skies)}')
print(f'Number of CPUs: {nCPU}')
print(f'Number of Ambient Bounces: {ab}')

# Creating the Octs dir if not existant
if not os.path.exists('Octs'):
    os.mkdir('Octs')

# Omitting rendered skies from the list
dirs = []

for i in os.listdir('Octs/'):
    if os.path.isdir('Octs/'+i):
        if not os.path.exists(f'Octs/{i}/done'):
            dirs.append(i)
            shutil.rmtree(f'Octs/{i}')
        else:
            try:
                skies.remove(f'{i}.sky')
            except ValueError:
                print(f'Warning! {i} not in list')

# Dividing skies into CPU clusters
divisionCPU = len(skies) // nCPU

skyDict = {}
for i in range(nCPU):
    skyDict['sky'+str(i)] = skies[i*divisionCPU:divisionCPU*(i+1)]

for i, sky in enumerate(skies[nCPU * divisionCPU:]):
    skyDict['sky'+str(i)].append(sky)
print(skyDict)
# print(len(list(skyDict.values())[5]))

# Changing the directory
os.chdir('Octs')

# Set the PATH variables
os.environ['PATH'] += os.pathsep + ':/usr/local/radiance/bin'
os.environ['RAYPATH'] = '.:/usr/local/radiance/lib'

# Setting empty list for bad HDRs
badDirs = []


# %%
def octMaker(sky, j):
    for i in sky:
        # Timing
        a = time.time()

        # make the oconv command
        oconv = f'oconv ../../skies/{i}.sky ../Geo.rad > {i}/render.oct'

        # make the rpict command
        render = f'X={imgDims}; Y={imgDims//2}; cnt $Y $X | rcalc -f ../2d360.cal ' + \
                 '-e "XD=$X;YD=$Y;X=5.0;Y=3.5;Z=1.2" | rtrace -n {nCores}' + \
                 '-x $X -y $Y -fac @../savedAB{ab}.opt {i}/render.oct > {i}/{i}.hdr'

        # Make the render matrix
        pvalue = f'pvalue -h -d -H -b -o {i}/{i}.hdr > {i}/{i}.txt'

        # Make the render directory
        os.mkdir(f'{i}')

        # Make the octree file
        os.system(oconv)

        # renders using rpict & remove the octree afterwards
        os.system(f'{render}')
        os.remove(f'{i}/render.oct')

        if os.system(pvalue):
            print(i, ' is bad image.')
            badDirs.append(i)
            os.remove(f'{i}/{i}.txt')
            continue
        else:
            print(i)
        with io.open(f'{i}/{i}.txt', 'r', encoding=None) as f:
            vector = np.loadtxt(f, delimiter='\n')
            MIN, MAX = vector.min(), vector.max()
            if MIN < minMaxDict[j][0]:
                minMaxDict[j][0] = MIN
            if MAX > minMaxDict[j][1]:
                minMaxDict[j][1] = MAX
            np.savetxt(f'{i}/{i}.gz', vector)
        os.remove(f'{i}/{i}.txt')

        # Finished statement
        os.system(f'touch {i}/done')
        os.system(f'echo "{time.time() - a:.2f}" > done')
        print(str(i + ' done!'))

    return 'Process #' + str(j) + ' is done!'


# %%
# The parallel processing module
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = [executor.submit(octMaker, list(
        skyDict['sky'+str(i)]), i) for i in range(nCPU)]

    for f in concurrent.futures.as_completed(results):
        print(f.result())

# %%
# minMax to numpy here
minMax = np.array(list(minMaxDict.values()))
MIN = minMax[:, 0].min()
MAX = minMax[:, 1].max()

with open('min-max.txt', 'w') as f:
    f.write(f'{MIN}, {MAX}')

with open('badDirs.txt', 'w') as f:
    # [f.write(i + ',') for i in badDirs]
    f.write(', '.join(badDirs))
