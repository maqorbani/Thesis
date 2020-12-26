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
        xA = xA.reshape(1, 1, 144, 256)
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
