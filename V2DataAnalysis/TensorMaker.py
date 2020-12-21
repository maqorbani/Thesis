'''
The results.txt extracted from the k-means step, should be placed in the root
of this directory as an indicator of X samples.
'''
# %%
import os
import numpy as np
import matplotlib.pyplot as plt

# %%
alt = np.loadtxt('Altitude.txt')  # Sun altitude
azi = np.loadtxt('Azimuth.txt') - 180  # Sun azimuth
dire = np.loadtxt('dirRad.txt')  # Sun direct radiation
dif = np.loadtxt('difHorRad.txt')  # Sky diffuse radiation
key = np.loadtxt('key.txt', dtype='str')  # Hour of year for each key
selKeys = np.loadtxt('results250.txt')  # K-means selected keys
selKeys = [int(i) for i in selKeys]
assert (len(alt) == len(azi) == len(dire) == len(dif) == len(key))
TheTuple = (3.945, 0.0, 8.969, 0.06066)

# %%


def TensorMaker(indices):
    '''
    Creates the tensor ready for the DL model training and testing
    The images required for this function are 144 by 256 pixels
    This function assumes that the previous cells has have executed
    therefore alt,azi,dire,dif and key are available in the memory
    n is the number of samples in the desired tensor
    '''
    n = len(indices)
    tnsr = np.zeros((n, 36864, 9))

    # X, Y
    tnsr[:, :, 0] = np.array(list(range(256))*144)
    tnsr[:, :, 1] = np.array(list(range(144))*256).reshape(256, 144).T.ravel()
    # Al, Az, Dir, Dif
    for i, x in enumerate(indices):
        tnsr[i, :, 2] = alt[x]
        tnsr[i, :, 3] = azi[x]
        tnsr[i, :, 4] = dire[x]
        tnsr[i, :, 5] = dif[x]
        tnsr[:, :, 6] += np.loadtxt(f'ab4/{key[x]}/{key[x]}.gz')  # Sum of all
        tnsr[i, :, 7] = np.loadtxt(f'ab0/{key[x]}/{key[x]}.gz')  # ab0
        tnsr[i, :, 8] = np.loadtxt(f'ab4/{key[x]}/{key[x]}.gz')  # ab4
    tnsr[:, :, 6] / n

    tnsr = tnsr.astype('float32')
    tnsr[:, :, :-2] = minMaxScale(tnsr[:, :, :-2])
    tnsr[:, :, -2:] = forceMinMax(tnsr[:, :, -2:], TheTuple)

    return tnsr


def minMaxScale(tnsr):
    for i in range(tnsr.shape[-1]):
        tnsr[:, :, i] = (tnsr[:, :, i]-tnsr[:, :, i].min()) / \
            (tnsr[:, :, i].max() - tnsr[:, :, i].min())

    return tnsr


def forceMinMax(tnsr, Tuples):
    ab0MAX, ab0min, ab4MAX, ab4min = Tuples

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
train = TensorMaker(selKeys)
np.save('train.npy', train)

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
test = TensorMaker(choice)
np.save('test_random.npy', test)

# %%
