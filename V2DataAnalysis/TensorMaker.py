'''
The results.txt extracted from the k-means step, should be placed in the
data folder at the root of this directory representing keys of X samples.
'''
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# %%
features = {
    "AVGMap": True,
    "STDmap": False,
    "NormalMap": True,
    "DepthMap": False,
    "ReflectionMap": False,
    "AOmap": True,
    "View_Number": 5,
    "Number of samples": 250,
    "Number of test": 250,
    "Transfer Learning": [25, 125, 25]
}

m = features["Number of samples"]
mTest = features["Number of test"]
View = features["View_Number"]

fileName = ""
fileName = fileName + '-AVG' if not features["AVGMap"] else fileName
fileName = fileName + '-STD' if features["STDmap"] else fileName
fileName = fileName + '-NM' if features["NormalMap"] else fileName
fileName = fileName + '-D' if features["DepthMap"] else fileName
fileName = fileName + '-R' if features["ReflectionMap"] else fileName
fileName = fileName + '-AO' if features["AOmap"] else fileName

n_axes = 6

if features["AVGMap"]:
    avg = n_axes
    n_axes += 1
if features["STDmap"]:
    std = n_axes
    n_axes += 1
if features["NormalMap"]:
    nrml = n_axes
    n_axes += 3
if features["DepthMap"]:
    dpth = n_axes
    n_axes += 1
if features["ReflectionMap"]:
    rflc = n_axes
    n_axes += 1
if features["AOmap"]:
    aomp = n_axes
    n_axes += 1

n_axes += 2

# %%
alt = np.loadtxt('data/Altitude.txt')          # Sun altitude
azi = np.loadtxt('data/Azimuth.txt') - 180     # Sun azimuth
dire = np.loadtxt('data/dirRad.txt')           # Sun direct radiation
dif = np.loadtxt('data/difHorRad.txt')         # Sky diffuse radiation
key = np.loadtxt('data/key.txt', dtype='str')  # Hour of year for each key
selKeys = np.loadtxt(f'data/results{m}.txt')   # K-means selected keys
selKeys = [int(i) for i in selKeys]
assert (len(alt) == len(azi) == len(dire) == len(dif) == len(key))

with open(f'../V{View}DataAnalysis/ab0/min-max.txt', 'r') as f:
    ab0 = [float(i) for i in f.read().split(', ')]

with open(f'../V{View}DataAnalysis/ab4/min-max.txt', 'r') as f:
    ab4 = [float(i) for i in f.read().split(', ')]

TheTuple = ab0 + ab4

if features["NormalMap"]:
    normalMap = Image.open(
        f'../SceneRefrences/V{View}Normal.jpg').resize((256, 144), 1)
    normalMap = np.asarray(normalMap).reshape(-1, 3) / 255

if features["DepthMap"]:
    depthMap = Image.open(
        f'../SceneRefrences/V{View}Depth.jpg').resize((256, 144), 1)
    depthMap = np.asarray(depthMap)[:, :, 0].reshape(-1) / 255

if features["ReflectionMap"]:
    reflectionMap = Image.open(
        f'../SceneRefrences/V{View}Reflection.jpg').resize((256, 144), 1)
    reflectionMap = np.asarray(reflectionMap)[:, :, 0].reshape(-1) / 255

if features["AOmap"]:
    AOmap = Image.open(
        f'../SceneRefrences/V{View}AO.jpg').resize((256, 144), 1)
    AOmap = np.asarray(AOmap)[:, :, 0].reshape(-1) / 255

# %%


def TensorMaker(indices, TheTuple):
    '''
    Creates the tensor ready for the DL model training and testing
    The images required for this function are 144 by 256 pixels
    This function assumes that the previous cells has have executed
    therefore alt,azi,dire,dif and key are available in the memory
    n is the number of samples in the desired tensor
    '''
    n = len(indices)
    tnsr = np.zeros((n, 36864, n_axes))

    #                                                             # X, Y
    tnsr[:, :, 0] = np.array(list(range(256))*144)
    tnsr[:, :, 1] = np.array(list(range(144))*256).reshape(256, 144).T.ravel()

    for i, x in enumerate(indices):
        tnsr[i, :, 2] = alt[x]                                    # altitude
        tnsr[i, :, 3] = azi[x]                                    # azimuth
        tnsr[i, :, 4] = dire[x]                                   # direct
        tnsr[i, :, 5] = dif[x]                                    # diffuse

        #                                                         # AB0, Ab4
        tnsr[i, :, -2] = \
            np.loadtxt(f'../V{View}DataAnalysis/ab0/{key[x]}/{key[x]}.gz')
        tnsr[i, :, -1] = \
            np.loadtxt(f'../V{View}DataAnalysis/ab4/{key[x]}/{key[x]}.gz')

    if features["AVGMap"]:                                        # Average
        tnsr[:, :, avg] = tnsr[:, :, -1].sum(axis=0) / n
    if features["STDmap"]:                                        # STD
        tnsr[:, :, std] = tnsr[:, :, -1].std(axis=0)
    if features["NormalMap"]:                                     # Normal map
        tnsr[:, :, nrml:nrml + 3] = normalMap
    if features["DepthMap"]:                                      # Depth map
        tnsr[:, :, dpth] = depthMap
    if features["ReflectionMap"]:                                 # Reflection
        tnsr[:, :, rflc] = reflectionMap
    if features["AOmap"]:                                         # AO map
        tnsr[:, :, aomp] = AOmap

    tnsr = tnsr.astype('float32')
    tnsr[:, :, :6 + features["AVGMap"] + features["STDmap"]] = \
        minMaxScale(tnsr[:, :, :6 + features["AVGMap"] + features["STDmap"]])
    tnsr[:, :, -2:] = forceMinMax(tnsr[:, :, -2:], TheTuple)

    return tnsr


def minMaxScale(tnsr):
    for i in range(tnsr.shape[-1]):
        tnsr[:, :, i] = (tnsr[:, :, i]-tnsr[:, :, i].min()) / \
            (tnsr[:, :, i].max() - tnsr[:, :, i].min())

    return tnsr


def forceMinMax(tnsr, Tuples):
    ab0min, ab0MAX, ab4min, ab4MAX = Tuples

    tnsr[:, :, 0] = (tnsr[:, :, 0] - ab0min) / (ab0MAX - ab0min)
    tnsr[:, :, 1] = (tnsr[:, :, 1] - ab4min) / (ab4MAX - ab4min)
    return tnsr


def minMaxFinder():
    ab0MAX = 0
    ab0min = 10000
    for i in os.listdir('ab0'):
        file = np.loadtxt(f'ab0/{i}/{i}.gz')
        if file.max() > ab0MAX:
            ab0MAX = file.max()
        if file.min() < ab0min:
            ab0min = file.min()

    ab4MAX = 0
    ab4min = 10000
    for i in os.listdir('ab4'):
        file = np.loadtxt(f'ab4/{i}/{i}.gz')
        if file.max() > ab4MAX:
            ab4MAX = file.max()
        if file.min() < ab4min:
            ab4min = file.min()

        return ab0MAX, ab0min, ab4MAX, ab4min


# %%
train = TensorMaker(selKeys, TheTuple)
# np.save('data/train' + fileName + '.npy', train)

# %%
testList = list(range(4141))
for i in selKeys:
    testList.remove(i)

# %%
# test = TensorMaker(testList)

# %%
choice = np.random.choice(testList, mTest)

plt.scatter(dire, dif, c='grey', s=2)
plt.scatter(dire[choice], dif[choice], c='red', s=10)
plt.show()

plt.scatter(azi, alt, c='grey', s=2)
plt.scatter(azi[choice], alt[choice], c='red', s=10)
plt.show()

# %%
test = TensorMaker(choice, TheTuple)
# np.save('data/test_random' + fileName + '.npy', test)

# %%
np.savez_compressed(f'../V{View}DataAnalysis/data/data' +
                    fileName+'.npz', train=train, test=test)

# %%
a, b, c = features["Transfer Learning"]
TLsamples = np.arange(a, b, c)

choice = np.random.choice(range(4141), mTest)
test = TensorMaker(choice, TheTuple)

for i in TLsamples:
    selKeys = np.loadtxt(f'data/results{i}.txt')
    selKeys = [int(i) for i in selKeys]
    train = TensorMaker(selKeys, TheTuple)
    np.savez_compressed(f'../V{View}DataAnalysis/data/data' +
                        fileName+f'-{i}.npz', train=train, test=test)
