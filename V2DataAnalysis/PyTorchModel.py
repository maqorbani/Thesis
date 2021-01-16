# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, n_features, device):
        super(Model, self).__init__()
        self.AAconv = nn.Conv2d(1, 400, 1)

        self.n_features = n_features
        self.device = device

        self.BBconv1 = nn.Conv2d(n_features - 1, 600, 1)
        self.BBconv2 = nn.Conv2d(600, 600, 1)
        self.BBconv3 = nn.Conv2d(600, 600, 1)
        self.BBconv4 = nn.Conv2d(600, 600, 1)

        self.CCconv = nn.Conv2d(1000, 600, 1)
        self.out = nn.Conv2d(600, 1, 1)

    def forward(self, x):
        x = x.to(self.device)

        # (36864, 1) the sunpatch images, reshaping for avoiding a size 1 array
        xA = x[-1, :, :]
        xA = xA.reshape(1, 1, xA.shape[-2], xA.shape[-1])
        # (36864, 7) other than sunpatch
        xB = x[:-1, :, :].unsqueeze(0)
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


class Model_2(nn.Module):
    def __init__(self, n_features, device):
        super(Model_2, self).__init__()

        self.n_features = n_features
        self.device = device

        self.AAconv = nn.Conv2d(1, 400, 1)
        self.AOconv = nn.Conv2d(1, 600, 1)

        self.BBconv1 = nn.Conv2d(n_features - 2, 600, 1)
        self.BBconv2 = nn.Conv2d(1200, 600, 1)
        self.BBconv3 = nn.Conv2d(600, 600, 1)
        self.BBconv4 = nn.Conv2d(600, 600, 1)
        self.BBconv5 = nn.Conv2d(600, 600, 1)
        self.BBconv6 = nn.Conv2d(600, 600, 1)

        self.CCconv = nn.Conv2d(1000, 600, 1)
        self.out = nn.Conv2d(600, 1, 1)

        # self.m = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.to(self.device)

        # (1, 144, 256) sunpatch image, reshaping for avoiding a size 1 array
        xA = x[-1, :, :]
        xA = xA.reshape(1, 1, xA.shape[-2], xA.shape[-1])  # (1, 1, 144, 256)
        xO = x[-2, :, :]
        xO = xO.reshape(1, 1, xO.shape[-2], xO.shape[-1])  # (1, 1, 144, 256)
        # (7, 144, 256) other than sunpatch
        xB = x[:-2, :, :].unsqueeze(0)                     # (1, n, 144, 256)
        del x

        xA = F.relu(self.AAconv(xA))

        xO = F.relu(self.AOconv(xO))

        xB = F.relu(self.BBconv1(xB))
        xB = torch.cat([xB, xO], dim=1)
        del xO
        xB = F.relu(self.BBconv2(xB))
        xB = F.relu(self.BBconv3(xB))
        xB = F.relu(self.BBconv4(xB))
        xB = F.relu(self.BBconv5(xB))
        xB = F.relu(self.BBconv6(xB))

        xB = torch.cat([xA, xB], dim=1)
        del xA
        xB = F.relu(self.CCconv(xB))
        xB = F.relu(self.out(xB))

        return xB


class Model_3(nn.Module):
    def __init__(self, n_features, device):
        super(Model_3, self).__init__()

        self.n_features = n_features
        self.device = device

        self.AAconv = nn.Conv2d(1, 400, 1)
        self.AOconv = nn.Conv2d(1, 600, 1)

        self.BBconv1 = nn.Conv2d(n_features - 2, 600, 1)
        self.BBconv2 = nn.Conv2d(600, 600, 1)
        self.BBconv3 = nn.Conv2d(1200, 600, 1)
        self.BBconv4 = nn.Conv2d(600, 600, 1)
        self.BBconv5 = nn.Conv2d(600, 600, 1)
        self.BBconv6 = nn.Conv2d(600, 600, 1)

        self.CCconv = nn.Conv2d(1000, 600, 1)
        self.out = nn.Conv2d(600, 1, 1)

        # self.m = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.to(self.device)

        # (1, 144, 256) sunpatch image, reshaping for avoiding a size 1 array
        xA = x[-1, :, :]
        xA = xA.reshape(1, 1, xA.shape[-2], xA.shape[-1])  # (1, 1, 144, 256)
        xO = x[-2, :, :]
        xO = xO.reshape(1, 1, xO.shape[-2], xO.shape[-1])  # (1, 1, 144, 256)
        # (7, 144, 256) other than sunpatch
        xB = x[:-2, :, :].unsqueeze(0)                     # (1, n, 144, 256)
        del x

        xA = F.relu(self.AAconv(xA))

        xO = F.relu(self.AOconv(xO))

        xB = F.relu(self.BBconv1(xB))
        xB = F.relu(self.BBconv2(xB))
        xB = torch.cat([xB, xO], dim=1)
        del xO
        xB = F.relu(self.BBconv3(xB))
        xB = F.relu(self.BBconv4(xB))
        xB = F.relu(self.BBconv5(xB))
        xB = F.relu(self.BBconv6(xB))

        xB = torch.cat([xA, xB], dim=1)
        del xA
        xB = F.relu(self.CCconv(xB))
        xB = F.relu(self.out(xB))

        return xB
