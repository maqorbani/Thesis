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
    "NormalMap": True,
    "DepthMap": True,
    "ReflectionMap": False
}

n_axes = 9
n_axes = n_axes + 3 if features["NormalMap"] else n_axes
n_axes = n_axes + 1 if features["DepthMap"] else n_axes
n_axes = n_axes + 1 if features["ReflectionMap"] else n_axes

# %%
alt = np.loadtxt('data/Altitude.txt')          # Sun altitude
azi = np.loadtxt('data/Azimuth.txt') - 180     # Sun azimuth
dire = np.loadtxt('data/dirRad.txt')           # Sun direct radiation
dif = np.loadtxt('data/difHorRad.txt')         # Sky diffuse radiation
key = np.loadtxt('data/key.txt', dtype='str')  # Hour of year for each key
selKeys = np.loadtxt('data/results250.txt')    # K-means selected keys
selKeys = [int(i) for i in selKeys]
assert (len(alt) == len(azi) == len(dire) == len(dif) == len(key))

with open('ab0/min-max.txt', 'r') as f:
    ab0 = [float(i) for i in f.read().split(', ')]

with open('ab4/min-max.txt', 'r') as f:
    ab4 = [float(i) for i in f.read().split(', ')]

TheTuple = ab0 + ab4

if features["NormalMap"]:
    normalMap = Image.open(
        '../SceneRefrences/V2Normal.jpg').resize((256, 144), 1)
    normalMap = np.asarray(normalMap).reshape(-1, 3) / 255

if features["DepthMap"]:
    depthMap = Image.open(
        '../SceneRefrences/V2Depth.jpg').resize((256, 144), 1)
    depthMap = np.asarray(depthMap)[:, :, 0].reshape(-1) / 255

if features["ReflectionMap"]:
    reflectionMap = Image.open(
        '../SceneRefrences/V2Reflection.jpg').resize((256, 144), 1)
    reflectionMap = np.asarray(reflectionMap)[:, :, 0].reshape(-1) / 255

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
        tnsr[:, :, 6] += np.loadtxt(f'ab4/{key[x]}/{key[x]}.gz')  # Sum (AVG)

        if features["NormalMap"]:
            tnsr[i, :, 7:10] = normalMap                          # Nomral map
        if features["DepthMap"]:
            tnsr[i, :, 10] = depthMap                             # Depth map
        if features["ReflectionMap"]:
            tnsr[i, :, 11] = reflectionMap                        # Reflection
        tnsr[i, :, -2] = np.loadtxt(f'ab0/{key[x]}/{key[x]}.gz')  # ab0
        tnsr[i, :, -1] = np.loadtxt(f'ab4/{key[x]}/{key[x]}.gz')  # ab4
    tnsr[:, :, 6] / n                                             # Average

    tnsr = tnsr.astype('float32')
    tnsr[:, :, :7] = minMaxScale(tnsr[:, :, :7])
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
np.save('train-NM-D.npy', train)

# %%
testList = list(range(4141))
for i in selKeys:
    testList.remove(i)

# %%
# test = TensorMaker(testList)

# %%
choice = np.random.choice(testList, 400)

# %%
plt.scatter(dire, dif, c='grey', s=2)
plt.scatter(dire[choice], dif[choice], c='red', s=10)
plt.show()

plt.scatter(azi, alt, c='grey', s=2)
plt.scatter(azi[choice], alt[choice], c='red', s=10)
plt.show()

# %%
test = TensorMaker(choice, TheTuple)
np.save('test_random-NM-D.npy', test)
