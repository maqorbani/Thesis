# %%
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from GPUtil import showUtilization as gpu_usage

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epoch = 10
batch = 8

x_train = np.load("train.npy")
x_test = np.load("test_random.npy")

m = x_train.shape[0]
mTest = x_test.shape[0]
# %%
# Transforming the data into torch tensors.
x_train, y_train = torch.tensor(x_train[:, :, :8]), \
    torch.tensor(x_train[:, :, -1])
x_test, y_test = torch.tensor(x_test[:, :, :8]), \
    torch.tensor(x_test[:, :, -1])

# relocation of the channel axes
x_train = np.transpose(x_train, [0, 2, 1]).reshape(-1, 8, 144, 256)
x_test = np.transpose(x_test, [0, 2, 1]).reshape(-1, 8, 144, 256)

# %%


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.AAconv = nn.Conv2d(1, 400, 1)

        self.BBconv1 = nn.Conv2d(7, 600, 1)
        self.BBconv2 = nn.Conv2d(600, 600, 1)
        self.BBconv3 = nn.Conv2d(600, 600, 1)
        self.BBconv4 = nn.Conv2d(600, 600, 1)

        self.CCconv = nn.Conv2d(1000, 600, 1)
        self.out = nn.Conv2d(600, 1, 1)

    def forward(self, x):
        x = x.to(device)

        # (36864, 1) the sunpatch images, reshaping for avoiding a size 1 array
        xA = x[7, :, :]
        xA = xA.reshape(1, 1, xA.shape[-2], xA.shape[-1])
        # (36864, 7) other than sunpatch
        xB = x[:7, :, :].unsqueeze(0)
        del x

        xA = F.relu(self.AAconv(xA))

        xB = F.relu(self.BBconv1(xB))
        xB = F.relu(self.BBconv2(xB))
        xB = F.relu(self.BBconv3(xB))
        xB = F.relu(self.BBconv4(xB))

        xB = torch.cat([xA, xB], dim=1)
        del xA
        xB = F.relu(self.CCconv(xB))
        xB = F.relu(self.out(xB))

        return xB


# %%
model = Model()
model.to(device)

print(model)
gpu_usage()

# %%
with torch.no_grad():
    out = model(x_train[3, :, :, :])
    plt.imshow(out.to("cpu").numpy().reshape(144, -1))
    plt.show()

# %%
torch.cuda.empty_cache()
gpu_usage()

# %%
epochLoss = []
criterion = nn.MSELoss()
testLoss = []

epochLossBatch = []
testLossBatch = []

# %%
optimizer = optim.Adam(model.parameters(), 0.00000005)
# model.zero_grad()   # zero the gradient buffe/rs

# %%

epochPercent = 0  # Dummy variable, just for printing purposes
# Model.train model.eval >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
for i in range(epoch*m):
    target = y_train[i % m, :].reshape(-1, 1)  # avoiding 1D array
    x = x_train[i % m, :, :, :]
    output = model(x).cpu().reshape(-1, 1)
    loss = criterion(output, target)
    loss.backward()

    epochLoss.append(loss.item())
    epochLossBatch.append(epochLoss[-1])

    with torch.no_grad():  # Calculating the test-set loss
        test_target = y_test[i % m, :].reshape(-1, 1)
        xTe = x_test[i % m, :, :, :]
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
        print('\n', "-->>Train>>", sum(epochLossBatch)/m)
        print("-->>Test>>", sum(testLossBatch)/mTest)
        epochLossBatch = []
        testLossBatch = []

# %%
plt.plot(np.log10(epochLoss))
plt.plot(np.log10(testLoss))

a = np.array(epochLoss)
for i in range(int(len(epochLoss) / m)):
    a[i*m:((i+1)*m)] = a[i*m:((i+1)*m)].mean()
plt.plot(np.log10(a), lw=4)

plt.show()
# %%
number = 12
with torch.no_grad():
    out = model(x_test[number, :, :]).to(
        "cpu").numpy().reshape(144, -1)+0.01
    T = y_test[number, :].to("cpu").numpy().reshape(144, -1)+0.01
# plt.imshow((out.to("cpu").detach().numpy().reshape(144, -1)))
# plt.show()

# Plotting both prediction and target images
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(40, 10))
ax1.imshow(np.log10(out))
ax1.title.set_text(f'{number}\nprediction')
ax2.imshow(np.log10(T))
ax2.title.set_text('ground_truth')
ax3.imshow(np.abs(out-T), vmax=0.2)
ax3.title.set_text('difference')
# plt.imshow((y_train[5, :].to("cpu").detach().numpy().reshape(144, -1))
#            - (out.to("cpu").detach().numpy().reshape(144, -1)), vmax=0.67)
plt.show()

# %%
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(7, 10))
ax1.hist((out).ravel(), bins=50)
ax1.title.set_text('y_hat')
ax2.hist((T).ravel(), bins=50)
ax2.title.set_text('ground_truth')
ax3.hist(np.abs(out-T).ravel(), bins=50)
ax3.title.set_text('difference')

plt.show()
# %%
torch.save(model.state_dict(), 'ConvModel.pth')

# %%
model.load_state_dict(torch.load("Model.pth"))
