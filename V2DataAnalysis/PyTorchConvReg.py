# %%
import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.colors import LogNorm
from GPUtil import showUtilization as gpu_usage
from sklearn.utils import shuffle
import cv2

from piq import psnr, ssim

# %%
Dictionary = {
    'epoch': 5,
    'batch': 8,
    'dataset': '-NM',
    'Model_Arch': 'UNet',
    'View #': 5,
    'Loss': 'MSE',               # Loss: MSE or MSE+RER
    'avg_shuffle': False,        # Shuffle mode
    'avg_division': 50,          # For shuffle mode only
    'transfer learning': True,   # TL mode
    '# samples': 75,            # For transfer Learning only
    '# NeighborPx': 1            # For model 3 and 4 px neighborhood
}

arch = Dictionary['Model_Arch']
theLoss = True if Dictionary['Loss'] == 'MSE+RER' else False
divAvg = Dictionary['avg_division']
pxNeighbor = Dictionary['# NeighborPx']
blnr = Dictionary['Bilinear']

if arch == 'UNet':
    from unet import UNet as Model              # UNet model
elif arch == 'DenseNet':
    from PyTorchModel import DenseNet as Model  # DenseNet model
elif arch == 1:
    from PyTorchModel import Model              # As proposed in the paper
elif arch == 2:
    from PyTorchModel import Model_2 as Model   # Has a branch for AO-map
elif arch == 3:
    from PyTorchModel import Model_3 as Model   # Has a branch for AO like #2
elif arch == 4:
    from PyTorchModel import Model_4 as Model   # Does not have extra branch
elif arch == 5:
    from PyTorchModel import Model_5 as Model   # ConvNet 3*3 for NM only
else:
    from PyTorchModel import Model_6 as Model   # The inception nwtwork idea

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epoch = Dictionary['epoch']
batch = Dictionary['batch']

data_set = Dictionary['dataset']                             # Data-sets
View = Dictionary['View #']

TLmode = Dictionary['transfer learning']
mTL = Dictionary['# samples']

if not TLmode:
    x_train = np.load(f'../V{View}DataAnalysis/data/data' +
                      data_set + '.npz')['train']
    x_test = np.load(f'../V{View}DataAnalysis/data/data' +
                     data_set + '.npz')['test']
else:
    x_train = np.load(f'../V{View}DataAnalysis/data/data' +
                      data_set + f'-{mTL}.npz')['train']
    x_test = np.load(f'../V{View}DataAnalysis/data/data' +
                     data_set + f'-{mTL}.npz')['test']


n_features = x_train.shape[-1] - 1
m = x_train.shape[0]
mTest = x_test.shape[0]

modelArgs = [n_features, 1, device, blnr] if arch == 'UNet' \
    else [n_features, device]
if arch in (3, 4):
    modelArgs.append(pxNeighbor)

# %%
# Transforming the data into torch tensors.
x_train, y_train = torch.tensor(x_train[:, :, :n_features]), \
    torch.tensor(x_train[:, :, -1])
x_test, y_test = torch.tensor(x_test[:, :, :n_features]), \
    torch.tensor(x_test[:, :, -1])

# relocation of the channel axes
x_train = np.transpose(x_train, [0, 2, 1]).reshape(-1, n_features, 144, 256)
x_test = np.transpose(x_test, [0, 2, 1]).reshape(-1, n_features, 144, 256)

# %%
# Load the model from PyTorchModel.py to the GPU
model = Model(*modelArgs)
model.to(device)

print(model)
gpu_usage()
print('\nNumber Of parameters:', sum(p.numel() for p in model.parameters()))

# For transfer learning model load
if TLmode:
    learnedView = 2
    model.load_state_dict(torch.load(
        f'../V{learnedView}DataAnalysis/models/' +
        f'ConvModel{data_set}-{arch}.pth'))

# %%
with torch.no_grad():
    out = model(x_train[3, :, :, :])
    plt.imshow(out.to("cpu").numpy().reshape(144, -1))
    plt.show()

torch.cuda.empty_cache()
gpu_usage()


# %%
def rer_loss(output, target):
    loss = torch.sqrt(torch.sum((output - target)**2) / torch.sum(target**2))
    return loss

# %%


def get_minMax(View):
    with open(f'../V{View}DataAnalysis/ab4/min-max.txt', 'r') as f:
        minMax = [float(i) for i in f.read().split(', ')]
    return np.log10(np.array(minMax))


def revert_HDR(HDR, minMax):
    HDR = HDR * (minMax[1] - minMax[0]) + minMax[0]
    HDR = np.power(10, HDR)
    return HDR


def predict_HDR_write(x, View, m=None):
    if m is None:
        m = x.shape[0]

    if len(x.shape) == 3:  # Check for one sample input
        x = x.unsqueeze(0)

    key = np.loadtxt('data/key.txt', dtype='str')  # Hour of year for each key
    selKeys = np.loadtxt(f'data/results{m}.txt')   # K-means selected keys
    selKeys = [int(i) for i in selKeys]

    minMax = get_minMax(View)

    with torch.no_grad():
        for i in range(x.shape[0]):
            out = model(x[i]).cpu().numpy().reshape(144, 256)
            out = revert_HDR(out, minMax)
            date_time = key[selKeys[i]]
            cv2.imwrite(f'ab4/{date_time}/{date_time}_2.HDR', out)
            print(date_time)


def get_date_time(index, m):
    key = np.loadtxt('data/key.txt', dtype='str')   # Hour of year for each key
    selKeys = np.loadtxt(f'data/results{m}.txt', dtype=int)  # K-means selected
    return key[selKeys[index]], selKeys[index]


# cv2.imwrite('out.HDR', revert_HDR(out, minMax))


# %%
criterion = nn.MSELoss()
epochLoss = []
testLoss = []

epochLossBatch = []
testLossBatch = []

if theLoss:
    def criterion(t, y): return nn.MSELoss()(t, y) + 10 * rer_loss(t, y)

# %%
if Dictionary['avg_shuffle']:
    minMax = np.load(
        f'../V{View}DataAnalysis/data/{data_set}-{m}-minMAX-key.npy')


def avg_shuffle(x, y):
    x, y = shuffle(x, y)
    for i in range(x.shape[0] // divAvg):
        x[i*divAvg: (i+1)*divAvg, 6] =\
            (y[i*divAvg: (i+1)*divAvg].sum(axis=0) / divAvg).reshape(144, -1)

    x[:, 6] = (x[:, 6] - minMax[6, 0]) / (minMax[6, 1] - minMax[6, 0])

    return x, y


# %%
optimizer = optim.Adam(model.parameters(), 0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 2,
                                                 verbose=True, threshold=1e-4,
                                                 cooldown=5)

# %%
# To change the learning rate
optimizer.param_groups[0]['lr'] = 0.00001

# %%
a = time.time()

epochPercent = 0  # Dummy variable, just for printing purposes
model.train()

for i in range(epoch*m):
    target = y_train[i % m, :].reshape(-1, 1)  # avoiding 1D array
    x = x_train[i % m, :, :, :]
    output = model(x).cpu().reshape(-1, 1)
    loss = criterion(output, target)
    loss.backward()

    epochLoss.append(loss.item())
    epochLossBatch.append(epochLoss[-1])

    with torch.no_grad():  # Calculating the test-set loss
        test_target = y_test[i % mTest, :].reshape(-1, 1)
        xTe = x_test[i % mTest, :, :, :]
        output = model(xTe).cpu().reshape(-1, 1)
        testLoss.append(criterion(output, test_target).item())
        testLossBatch.append(testLoss[-1])

    if i % batch == batch - 1:
        optimizer.step()
        model.zero_grad()

    if (i + 1) * 10 // m == epochPercent + 1:
        print("#", end='')
        epochPercent += 1

    if i % m == m - 1:
        epochLossBatchAvg = sum(epochLossBatch)/m
        print('\n', "-->>Train>>", epochLossBatchAvg)
        print("-->>Test>>", sum(testLossBatch)/m)
        epochLossBatch = []
        testLossBatch = []

        scheduler.step(epochLossBatchAvg)

        if Dictionary['avg_shuffle']:
            x_train, y_train = avg_shuffle(x_train, y_train)

print(f'\nIn {time.time() - a:.2f} Seconds')
# %%
plt.plot(np.log10(epochLoss))
plt.plot(np.log10(testLoss))

a = np.array(epochLoss)
for i in range(int(len(epochLoss) / m)):
    a[i*m:((i+1)*m)] = a[i*m:((i+1)*m)].mean()
plt.plot(np.log10(a), lw=4)

plt.show()
# %%
# model.eval()
number = 15
with torch.no_grad():
    out = revert_HDR(model(x_train[number, :, :]).to(
        "cpu").numpy().reshape(144, -1), get_minMax(View)) * 179
    T = revert_HDR(y_train[number, :].to("cpu").numpy().reshape(144, -1),
                   get_minMax(View)) * 179
# plt.imshow((out.to("cpu").detach().numpy().reshape(144, -1)))
# plt.show()

# Plotting both prediction and target images
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(40, 10))
im = ax1.imshow(np.log10(out), cmap='plasma', vmin=0, vmax=4)
ax1.title.set_text(f'{number}\nprediction')
ax2.imshow(np.log10(T), cmap='plasma', vmin=0, vmax=4)
ax2.title.set_text('ground_truth')
ax3.imshow(np.log10(np.abs(out-T)), cmap='plasma', vmin=0, vmax=4)
ax3.title.set_text('difference')

# fig.colorbar(im, ticks=[0.1, 0.3, 0.5, 0.7, 0.9])
# ax1.imshow(out, cmap='plasma',  norm=LogNorm(vmin=0.01, vmax=0.65))
# fig.colorbar(im, cax=axcolor, ticks=t, format='$%.2f$')
# plt.imshow((y_train[5, :].to("cpu").detach().numpy().reshape(144, -1))
#            - (out.to("cpu").detach().numpy().reshape(144, -1)), vmax=0.67)
plt.show()

# %%
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(7, 10))
ax1.hist((out).ravel(), bins=100, range=(0, 1))
ax1.title.set_text('y_hat')
ax2.hist((T).ravel(), bins=100, range=(0, 1))
ax2.title.set_text('ground_truth')
ax3.hist(np.abs(out-T).ravel(), bins=100, range=(0, 1))
ax3.title.set_text('difference')

plt.show()
# %%
answer = input("Are you sure that you want to save? [yes/any]")

if answer == 'yes':
    if not TLmode:
        torch.save(model.state_dict(),
                   f'../V{View}DataAnalysis/models/' +
                   f'ConvModel{data_set}-{arch}.pth')
    else:
        torch.save(model.state_dict(),
                   f'../V{View}DataAnalysis/models/' +
                   f'ConvModel{data_set}-{arch}-{mTL}.pth')


# %%
if not TLmode:
    model.load_state_dict(torch.load(
        f'../V{View}DataAnalysis/models/ConvModel{data_set}-{arch}.pth'))
else:
    model.load_state_dict(torch.load(
        f'../V{View}DataAnalysis/models/ConvModel{data_set}-{arch}-{mTL}.pth'))

# %%
# For transfer learning model load
learnedView = 2
model.load_state_dict(torch.load(
    f'../V{learnedView}DataAnalysis/models/ConvModel{data_set}-{arch}.pth'))

# %%
# Loss calculator over the train-test sets
a = time.time()

train_loss = []
test_loss = []

train_illum = []
test_illum = []

with torch.no_grad():
    for i in range(m):
        target = y_train[i, :].reshape(-1, 1)  # avoiding 1D array
        x = x_train[i, :, :, :]
        output = model(x).cpu().reshape(-1, 1)
        loss = criterion(output, target)
        train_loss.append(loss.item())
        train_illum.append(x[-1].mean().item())
    for i in range(mTest):
        target = y_test[i, :].reshape(-1, 1)  # avoiding 1D array
        x = x_test[i, :, :, :]
        output = model(x).cpu().reshape(-1, 1)
        loss = criterion(output, target)
        test_loss.append(loss.item())
        test_illum.append(x[-1].mean().item())

print(sum(train_loss)/m)
print(sum(test_loss)/mTest)

print(f'\nIn {time.time() - a:.2f} Seconds')
# %%
a = time.time()

train_ssim = []
test_ssim = []
train_psnr = []
test_psnr = []

with torch.no_grad():
    for i in range(m):
        target = y_train[i, :].reshape(1, 1, 144, 256)
        x = x_train[i, :, :, :]
        output = model(x).cpu()
        loss = ssim(target, output)
        train_ssim.append(loss.item())
        loss = psnr(target, output)
        train_psnr.append(loss.item())
    for i in range(mTest):
        target = y_test[i, :].reshape(1, 1, 144, 256)
        x = x_test[i, :, :, :]
        output = model(x).cpu()
        loss = ssim(target, output)
        test_ssim.append(loss.item())
        loss = psnr(target, output)
        test_psnr.append(loss.item())

print('SSIM')
print(sum(train_ssim)/m)
print(sum(test_ssim)/mTest)
print('PSNR')
print(sum(train_psnr)/m)
print(sum(test_psnr)/mTest)

print(f'\nIn {time.time() - a:.2f} Seconds')

# %%

T1 = revert_HDR(T, get_minMax(5))
out1 = revert_HDR(out, get_minMax(5))

plt.scatter(T1.ravel()*179, out1.ravel()*179, s=1)
plt.plot([0, T1.max()*179], [0, T1.max()*179], color='red')
plt.xlabel('Ground truth luminance value')
plt.ylabel('Prediced luminance value')

# %%
fig, ax1 = plt.subplots(1, figsize=(10, 7))
plt.hist(revert_HDR(x_train[:, :, -1].ravel(),
                    minMax)/143, bins=200, color='black')

# %%
# Log plot section
x = np.arange(0.00001, 3, 0.00001)
log = np.log10(x)
dLog = 1/(x*np.log(10))

fig, ax1 = plt.subplots(1, figsize=(10, 8))
ax1.plot(x, log, color='black', lw=3)
ax1.plot(x, dLog, color='black', ls='--')
# ax1.plot([-3, 3], [0, 0], color=[0.3,0.3,0.3])
# ax1.annotate('data hist', (0.05, 1))
ax1.set_xlim([-0.04, 1])
ax1.set_ylim([-3, 6])
ax1.grid(True)
ax1.legend(['Log(x)', 'd/dx (log(x))'], prop={'size': 30})
# %%
fig, ax1 = plt.subplots(1, figsize=(15, 3))
ax1.hist(revert_HDR(x_train[:, :, -1], get_minMax(2)
                    ).ravel()/143, bins=200, color='black')
ax1.set_xlim([0, 1])
fig.patch.set_visible(False)
ax1.axis('off')
plt.savefig('hist.png')

# %%
plt.style.use('seaborn')
plt.style.use('seaborn-talk')
plt.style.use('seaborn-colorblind')
# 'seaborn', 'seaborn-colorblind', 'seaborn-talk'
data = np.loadtxt('DiamondChartUNet.csv', delimiter=',')
params = ['Base Model', 'Normal map', 'Standard Deviation map',
          'Depth map', 'Reflectance map', 'UNet-512 + Normal map']

fig, ax = plt.subplots(1, figsize=(15, 15), subplot_kw=dict(polar=True))
# fig.set_color_cycle(sns.color_palette("mako", 5))
for i in range(6):
    ax.fill([0, np.pi*0.5, np.pi, np.pi*1.5],
            data[i], alpha=0.35, edgecolor='#000000', linewidth=1,
            zorder=1 if i == 5 else i)

ax.legend(params, loc='upper right')
ax.tick_params(labelsize=13)
for i in range(5):
    ax.plot([0, np.pi*0.5, np.pi, np.pi*1.5, 0], np.hstack((data[i], data[i, 0])),
            alpha=1, linewidth=0.2, color='black')
ax.grid(color='#AAAAAA')
ax.set_rgrids([0, 1], color='#FFFFFF')
ax.set_facecolor('#FFFFFF')
ax.spines['polar'].set_color('#222222')
# ax.tick_params(colors='#FFFFFF')
ax.set_thetagrids(np.degrees(
    [0, np.pi*0.5, np.pi, np.pi*1.5]), ['MSE', 'RER', 'SSIM', 'PSNR'])
# ax.plot([-1, 1], [0, 0], color=[0.6,0.6,0.6])
# ax.plot([0, 0], [-1, 1], color=[0.6,0.6,0.6])
# ax.annotate('Performance', [0, -1.05])
# ax.annotate('MSE', [1.02, 0])
# ax.annotate('SSIM', [0, 1.03])
# ax.annotate('PSNR', [-1.08, 0])
# plt.savefig('diamondChartUNet.png', dpi=150)
# %%
plt.style.use('seaborn')
plt.style.use('seaborn-talk')
plt.style.use('seaborn-colorblind')

data = np.loadtxt('DiamondChartSpeed.csv', delimiter=',')
params = ['Base Model', 'Normal map', 'Standard Deviation map',
          'Depth map', 'Reflectance map', 'UNet-512 + Normal map']

fig, ax = plt.subplots(1, figsize=(15, 15), subplot_kw=dict(polar=True))
# fig.set_color_cycle(sns.color_palette("mako", 5))
for i in range(6):
    ax.fill([np.pi*0.1, np.pi*0.5, np.pi*0.9, np.pi*1.3, np.pi*1.7],
            data[i], alpha=0.35, edgecolor='#000000', linewidth=1,
            zorder=0 if i == 5 else None)

ax.legend(params, loc='upper right')
ax.tick_params(labelsize=13)
for i in range(6):
    ax.plot([np.pi*0.1, np.pi*0.5, np.pi*0.9, np.pi*1.3, np.pi*1.7, np.pi*0.1],
            np.hstack((data[i], data[i, 0])), alpha=1, linewidth=0.2, color='black')
ax.grid(color='#AAAAAA')
ax.set_rgrids([0, 1], color='#FFFFFF')
ax.set_facecolor('#FFFFFF')
ax.spines['polar'].set_color('#222222')
# ax.tick_params(colors='#FFFFFF')
ax.set_thetagrids(np.degrees(
    [np.pi*0.1, np.pi*0.5, np.pi*0.9, np.pi*1.3, np.pi*1.7]), ['MSE', 'RER', 'SSIM', 'PSNR', 'Runtime'])
# ax.plot([-1, 1], [0, 0], color=[0.6,0.6,0.6])
# ax.plot([0, 0], [-1, 1], color=[0.6,0.6,0.6])
# ax.annotate('Performance', [0, -1.05])
# ax.annotate('MSE', [1.02, 0])
# ax.annotate('SSIM', [0, 1.03])
# ax.annotate('PSNR', [-1.08, 0])
plt.savefig('diamondChartRuntime.png', dpi=150)
# %%
# %%
# a = np.loadtxt('data/results16.txt')
alt = np.loadtxt('data/Altitude.txt')
azi = np.loadtxt('data/Azimuth.txt') - 180
dire = np.loadtxt('data/dirRad.txt')
dif = np.loadtxt('data/difHorRad.txt')
key = np.loadtxt('data/key.txt', dtype='str')

with torch.no_grad():
    for nu in a[:5]:
        out = model(x_train[nu, :, :]).to(
            "cpu").numpy().reshape(144, -1)
        T = y_train[nu, :].to("cpu").numpy().reshape(144, -1)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(
            30, 10), gridspec_kw={'wspace': 0.08, 'hspace': 0})
        im = ax1.imshow(out, cmap='plasma', vmin=0, vmax=0.9)
        ax1.title.set_text(
            f'{key[nu][:-2]}:00\ndirect: {int(dire[nu])} wh/m2\ndiffuse: {int(dif[nu])} wh/m2\n')
        ax2.imshow(T, cmap='plasma', vmin=0, vmax=0.9)
        # ax2.title.set_text('ground_truth')
        ax3.imshow(np.abs(out-T), cmap='plasma', vmin=0, vmax=0.9)
        # ax3.title.set_text('difference')
        # plt.savefig(f'../result/False_color/{nu}.png', dpi=100)

# %%
a = [156, 556, 744, 1032, 2081, 3439, 3573, 3363]
date = get_date_time(a, m)[0]
hoy = get_date_time(a, m)[1]

alt = np.loadtxt('data/Altitude.txt')
azi = np.loadtxt('data/Azimuth.txt') - 180
dire = np.loadtxt('data/dirRad.txt')
dif = np.loadtxt('data/difHorRad.txt')

with torch.no_grad():
    for index, nu in enumerate(a):
        out = revert_HDR(model(x_train[nu, :, :]).to(
            "cpu").numpy().reshape(144, -1), get_minMax(View)) * 179
        T = revert_HDR(y_train[nu, :].to("cpu").numpy().reshape(144, -1),
                       get_minMax(View)) * 179
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(
            30, 10), gridspec_kw={'wspace': 0.08, 'hspace': 0})
        im = ax1.imshow(np.log10(out), cmap='plasma', vmin=0, vmax=4)
        # ax1.title.set_text(
        #     f'{date[nu][:-2]}:00\ndirect: {int(dire[hoy[nu]])}' +
        #     f' wh/m2\ndiffuse: {int(dif[hoy[nu]])} wh/m2\n')
        ax2.imshow(np.log10(T), cmap='plasma', vmin=0, vmax=4)
        # ax2.title.set_text('ground_truth')
        ax3.imshow(np.log10(np.abs(out-T)), cmap='plasma', vmin=0, vmax=4)
        # ax3.title.set_text('difference')
        # fig.colorbar(im)
        # plt.savefig(f'../result/False_color/V2{hoy[index]}.png', dpi=300)
        print(T.max())
