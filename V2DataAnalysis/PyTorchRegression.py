# %%
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from GPUtil import showUtilization as gpu_usage

# from statistics import median

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

# %%


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.AAfc = nn.Linear(1, 600)

        self.BBfc1 = nn.Linear(7, 600)
        self.BBfc2 = nn.Linear(600, 600)
        self.BBfc3 = nn.Linear(600, 600)
        self.BBfc4 = nn.Linear(600, 600)

        self.CCfc = nn.Linear(1200, 600)
        self.Out = nn.Linear(600, 1)

    def forward(self, x):
        x = x.to(device)  # import the tensror to device

        # (26450, 1) the sunpatch images, reshaping for avoiding a size 1 array
        xA = x[:, 7].view(-1, 1)
        xA = F.relu(self.AAfc(xA))

        xB = x[:, :7]  # (26450, 7) other than sunpatch
        xB = F.relu(self.BBfc1(xB))
        xB = F.relu(self.BBfc2(xB))
        xB = F.relu(self.BBfc3(xB))
        xB = F.relu(self.BBfc4(xB))

        xB = torch.cat([xA, xB], dim=1)
        del xA
        xB = F.relu(self.CCfc(xB))
        xB = F.relu(self.Out(xB))

        return xB


# %%
model = Model()
model.to(device)

print(model)
gpu_usage()

# %%
'''
# Testing a tensor on the model
out = model(x_train[3, :, :])
# Setting the loss function
criterion = nn.MSELoss()
loss = criterion(out, y_train[3,:].reshape(-1, 1))
print(loss)
# Zeroing out the gradients
model.zero_grad()
loss.backward()
print(model.CCfc.weight.grad)
'''
# %%
gpu_usage()
with torch.no_grad():
    out = model(x_train[3, :, :])
    plt.imshow(out.to("cpu").numpy().reshape(144, -1))
    plt.show()

gpu_usage()

plt.imshow(y_train[3, :].reshape(144, -1))
plt.show()
gpu_usage()

torch.cuda.empty_cache()
gpu_usage()

# %%
epochLoss = []
criterion = nn.MSELoss()
testLoss = []

epochLossBatch = []
testLossBatch = []

# %%
optimizer = optim.Adam(model.parameters(), 0.0000003)
# model.zero_grad()   # zero the gradient buffe/rs

# %%

epochPercent = 0  # Dummy variable, just for printing purposes
# Model.train model.eval >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
for i in range(epoch*m):
    target = y_train[i % m, :].reshape(-1, 1)  # avoiding 1D array
    x = x_train[i % m, :, :]
    output = model(x).cpu()
    loss = criterion(output, target)
    loss.backward()

    epochLoss.append(loss.item())
    epochLossBatch.append(epochLoss[-1])

    with torch.no_grad():  # Calculating the test-set loss
        test_target = y_test[i % m, :].reshape(-1, 1)
        xTe = x_test[i % m, :, :]
        output = model(xTe).cpu()
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

    # model.zero_grad()   # zero the gradient buffers


# %%
'''
# Some creative stuff here
for i in range(int(epoch * m / batch)):
    mPrime = m + batch if (i+1)*batch % m == 0 else m
    target = y_train[(i*batch) % m:(i+1)*batch % mPrime, :].reshape(-1, 1)
    x = x_train[(i*batch) % m:(i+1)*batch % mPrime, :, :].reshape(-1, 8)
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()    # Does the update
    epochLoss.append(loss.item())
    test_target = y_test[(i*batch) % m:(i+1)*batch % mPrime, :].reshape(-1, 1)
    xTe = x_test[(i*batch) % m:(i+1)*batch % mPrime, :, :].reshape(-1, 8)
    output = model(xTe)
    testLoss.append(criterion(output, test_target).item())
'''

# %%
plt.plot(np.log10(epochLoss))
plt.plot(np.log10(testLoss))
plt.show()
# %%
number = 9
with torch.no_grad():
    out = model(x_test[number, :, :]).to(
        "cpu").numpy().reshape(144, -1)+0.01
    T = y_test[number, :].to("cpu").numpy().reshape(144, -1)+0.01
# plt.imshow((out.to("cpu").detach().numpy().reshape(144, -1)))
# plt.show()

# Plotting both prediction and target images
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(40, 10))
ax1.imshow(np.log10(out))
ax1.title.set_text('prediction')
ax2.imshow(np.log10(T))
ax2.title.set_text('ground_truth')
ax3.imshow((out-T), vmax=0.2)
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
# torch.save(model, 'Model')  # WARNING!!!!!!!!!!!!!!
torch.save(model.state_dict, 'Model.pth')

# %%
model = torch.load("Model")

# %%
testLoss = []
for i in range(y_test.size()[0]):
    target = y_test[i, :].reshape(-1, 1)
    x = x_test[i, :, :]
    output = model(x)
    testLoss.append(criterion(output, target).item())

print(sum(testLoss)/len(testLoss))


# %%


def analyze_results(sample_num):
    out = model(x_test[sample_num, :, :])
    out = np.rot90(out.to("cpu").detach().numpy().reshape(230, -1))
    fig = plt.figure(figsize=(15, 20))
    fig.add_subplot(3, 1, 1)
    plt.imshow(out, vmin=0, vmax=0.6)
    plt.title("Predicted")
    plt.colorbar()
    fig.add_subplot(3, 1, 2)
    target = np.rot90(y_test[sample_num, :].cpu().reshape(230, -1))
    plt.imshow(target, vmin=0, vmax=0.6)
    plt.title("Target")
    plt.colorbar()
    fig.add_subplot(3, 1, 3)
    plt.imshow(target-out, vmin=0, vmax=0.6)
    plt.title("difference")
    plt.colorbar()
    # plt.show()


analyze_results(testLoss.index(min(testLoss)))
sample_num = testLoss.index(min(testLoss))
# %%
out = model(x_train[sample_num, :, :])
out = np.rot90(out.to("cpu").detach().numpy().reshape(230, -1))
target = np.rot90(y_train[sample_num, :].cpu().reshape(230, -1))
print(np.max(np.abs(out-target)))
plt.imshow(np.abs(out-target))
