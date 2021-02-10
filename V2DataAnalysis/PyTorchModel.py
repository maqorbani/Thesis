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
    def __init__(self, n_features, device, pixel):
        super(Model_3, self).__init__()

        self.n_features = n_features
        self.device = device

        self.kernel = pixel*2 + 1
        self.padding = pixel

        self.AAconv = nn.Conv2d(1, 400, 1)
        self.AOconv = nn.Conv2d(1, 600, 1)

        self.BBconv1 = nn.Conv2d(
            n_features - 2, 600, self.kernel, padding=self.padding,
            padding_mode='reflect')
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


class Model_4(nn.Module):
    def __init__(self, n_features, device, pixel):
        super(Model_4, self).__init__()

        self.n_features = n_features
        self.device = device

        self.kernel = pixel*2 + 1
        self.padding = pixel

        self.AAconv = nn.Conv2d(1, 400, 1)

        self.BBconv1 = nn.Conv2d(
            n_features - 1, 600, self.kernel, padding=self.padding,
            padding_mode='reflect')
        self.BBconv2 = nn.Conv2d(600, 600, 1)
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

        # (7, 144, 256) other than sunpatch
        xB = x[:-1, :, :].unsqueeze(0)                     # (1, n, 144, 256)
        del x

        xA = F.relu(self.AAconv(xA))

        xB = F.relu(self.BBconv1(xB))
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


class Model_5(nn.Module):
    '''
    Passing the normal map through a convNet 3*(default) before
    concatenating it to the main channels
    '''

    def __init__(self, n_features, device, NMkernel=3):
        super(Model_5, self).__init__()
        self.n_features = n_features
        self.device = device
        self.NMkernel = NMkernel

        self.AAconv = nn.Conv2d(1, 400, 1)

        self.NMconv = nn.Conv2d(3, 1, self.NMkernel, padding=1)

        self.BBconv1 = nn.Conv2d(n_features - 3, 600, 1)
        self.BBconv2 = nn.Conv2d(600, 600, 1)
        self.BBconv3 = nn.Conv2d(600, 600, 1)
        self.BBconv4 = nn.Conv2d(600, 600, 1)

        self.CCconv = nn.Conv2d(1000, 600, 1)
        self.out = nn.Conv2d(600, 1, 1)

    def forward(self, x):
        x = x.to(self.device)

        NM = x[-4:-1, :, :].unsqueeze(0)
        NM = F.relu(self.NMconv(NM))

        # (36864, 1) the sunpatch images, reshaping for avoiding a size 1 array
        xA = x[-1, :, :]
        xA = xA.reshape(1, 1, xA.shape[-2], xA.shape[-1])
        # (36864, 7) other than sunpatch
        xB = x[:-4, :, :].unsqueeze(0)
        xB = torch.cat([xB, NM], dim=1)
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


class Model_6(nn.Module):
    '''
    Passing the normal map through an inception network
    before concatenating it to the main channels
    '''

    def __init__(self, n_features, device):
        super(Model_6, self).__init__()
        self.n_features = n_features
        self.device = device

        self.AAconv = nn.Conv2d(1, 400, 1)

        self.NMconv1 = nn.Conv2d(3, 3, 1, padding=0)
        self.NMconv3 = nn.Conv2d(3, 3, 3, padding=1)
        self.NMconv5 = nn.Conv2d(3, 3, 5, padding=2)

        self.BBconv1 = nn.Conv2d(n_features + 5, 600, 1)
        self.BBconv2 = nn.Conv2d(600, 600, 1)
        self.BBconv3 = nn.Conv2d(600, 600, 1)
        self.BBconv4 = nn.Conv2d(600, 600, 1)

        self.CCconv = nn.Conv2d(1000, 600, 1)
        self.out = nn.Conv2d(600, 1, 1)

    def forward(self, x):
        x = x.to(self.device)

        NM = x[-4:-1, :, :].unsqueeze(0)
        NM1 = F.relu(self.NMconv1(NM))
        NM3 = F.relu(self.NMconv3(NM))
        NM5 = F.relu(self.NMconv5(NM))
        del NM

        # (36864, 1) the sunpatch images, reshaping for avoiding a size 1 array
        xA = x[-1, :, :]
        xA = xA.reshape(1, 1, xA.shape[-2], xA.shape[-1])
        # (36864, 7) other than sunpatch
        xB = x[:-4, :, :].unsqueeze(0)
        xB = torch.cat([xB, NM1, NM3, NM5], dim=1)
        del x, NM1, NM3, NM5

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
