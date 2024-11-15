import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
dev = torch.device('cuda')

def apply(Fsom, FS, path_to_weights, batch_size=1):    
    # Fsom: NN by NT; fluorescence after neuropil subtraction and baselining
    # FS: sampling rate in Hz
    # path_to_weights: full path to weights
    # batch_size: can be increased if you have enough GPU RAM. 8 is 2x faster than 1, so not much gains. 

    # returns sp: firing rate estimate in spikes/s 

    # example call:  sp = udcnv.apply(Fcorrected, 30, 'E:/data/deconv/sim_right_flex.th', batch_size=1)

    n0 = 16
    nbase = [2, n0, 2*n0, 4*n0, 8*n0]

    net  = CPnet(nbase, 1, 3, conv_1D = True, style_on = True).to(dev)
    net.load_state_dict(torch.load(path_to_weights, weights_only=True))
    net.eval()

    ff = 8
    NN, NT = Fsom.shape
    Fdev         = torch.zeros((NN, 1, ff * ((NT-1)//ff + 1)), device = dev) 
    Fdev[:,0, :NT] = torch.from_numpy(Fsom).to(dev)
    Fdev[:,0,NT:] = Fdev[:,0, NT-1:NT]

    Fdev = Fdev - Fdev.mean(-1,keepdim=True)
    Fdev = Fdev / (Fdev**2).mean(-1,keepdim=True)**.5

    NT_pad = Fdev.shape[-1]
    sp = np.zeros((NN, NT), 'float32')
    for j in range(len(Fdev)//batch_size +1):
        Xtk = Fdev[j*batch_size:(j+1)*batch_size]
        
        Xtk_d = add_smooth(Xtk, np.array([FS]), sig = 0.1, dev = dev)   

        with torch.no_grad():
            y = net(Xtk_d)[0]
            y = 1e-2 + F.relu(y)
        sp[j*batch_size:(j+1)*batch_size] = y.cpu().numpy()[:,0,:NT] * 70/10

    return sp 

def add_smooth(Xtk, fs_target, sig = 0.1, dev = None):
    Xtk_d = torch.tile(Xtk, (1,2,1))

    if len(fs_target)==1:
        fs_target = fs_target * np.ones((Xtk_d.shape[0],))

    for j in range(len(Xtk)):
        ksamps = int(30  * sig * fs_target[j])
        kgauss = torch.exp( - torch.linspace(-3, 3, 2*ksamps+1, device = dev)**2).unsqueeze(0).unsqueeze(0)
        kgauss /= kgauss.sum()
        
        Xtk_d[j,1] = F.conv1d(Xtk_d[j:j+1,1:2], kgauss, padding = ksamps)
    return Xtk_d

"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
class CPnet(nn.Module):
    def __init__(self, nbase, nout, sz, mkldnn=False, conv_1D=False, max_pool=True,
                 diam_mean=30., style_on = True):
        super().__init__()
        self.nbase = nbase
        self.nout = nout
        self.sz = sz
        self.residual_on = True
        self.style_on = style_on
        self.concatenation = False
        self.conv_1D = conv_1D
        self.mkldnn = mkldnn if mkldnn is not None else False
        self.downsample = downsample(nbase, sz, conv_1D=conv_1D, max_pool=max_pool)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.upsample = upsample(nbaseup, sz, conv_1D=conv_1D)
        self.make_style = make_style(conv_1D=conv_1D)
        self.output = batchconv(nbaseup[0], nout, 1, conv_1D=conv_1D)
        self.diam_mean = nn.Parameter(data=torch.ones(1) * diam_mean,
                                      requires_grad=False)
        self.diam_labels = nn.Parameter(data=torch.ones(1) * diam_mean,
                                        requires_grad=False)


    def forward(self, data, style = None):

        if self.mkldnn:
            data = data.to_mkldnn()
        T0 = self.downsample(data)
        if style is None:
            if self.mkldnn:
                style = self.make_style(T0[-1].to_dense())
            else:
                style = self.make_style(T0[-1])
        style0 = style
        if not self.style_on:
            style = style * 0
        T1 = self.upsample(style, T0, self.mkldnn)
        T1 = self.output(T1)
        if self.mkldnn:
            T0 = [t0.to_dense() for t0 in T0]
            T1 = T1.to_dense()
        return T1, style0, T0
    
def batchconv(in_channels, out_channels, sz, conv_1D=False):
    conv_layer = nn.Conv1d if conv_1D else nn.Conv2d
    batch_norm = nn.BatchNorm1d if conv_1D else nn.BatchNorm2d
    return nn.Sequential(
        batch_norm(in_channels, eps=1e-5, momentum=0.05),
        nn.ReLU(inplace=True),
        conv_layer(in_channels, out_channels, sz, padding=sz // 2),
    )


def batchconv0(in_channels, out_channels, sz, conv_1D=False, batch = True):
    conv_layer = nn.Conv1d if conv_1D else nn.Conv2d
    batch_norm = nn.BatchNorm1d if conv_1D else nn.BatchNorm2d
    if batch: 
        mstep = nn.Sequential(
            batch_norm(in_channels, eps=1e-5, momentum=0.05),
            conv_layer(in_channels, out_channels, sz, padding=sz // 2),
            )
    else:
        mstep = nn.Sequential(            
            conv_layer(in_channels, out_channels, sz, padding=sz // 2),
            )

    return mstep


class resdown(nn.Module):

    def __init__(self, in_channels, out_channels, sz, conv_1D=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.proj = batchconv0(in_channels, out_channels, 1, conv_1D, batch = True)
        for t in range(4):
            if t == 0:
                self.conv.add_module("conv_%d" % t,
                                     batchconv(in_channels, out_channels, sz, conv_1D))
            else:
                self.conv.add_module("conv_%d" % t,
                                     batchconv(out_channels, out_channels, sz, conv_1D))

    def forward(self, x):
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x


class downsample(nn.Module):

    def __init__(self, nbase, sz, conv_1D=False, max_pool=True):
        super().__init__()
        self.down = nn.Sequential()
        if max_pool:
            self.maxpool = nn.MaxPool1d(2, stride=2) if conv_1D else nn.MaxPool2d(
                2, stride=2)
        else:
            self.maxpool = nn.AvgPool1d(2, stride=2) if conv_1D else nn.AvgPool2d(
                2, stride=2)
        for n in range(len(nbase) - 1):
            self.down.add_module("res_down_%d" % n,
                                 resdown(nbase[n], nbase[n + 1], sz, conv_1D))

    def forward(self, x):
        xd = []
        for n in range(len(self.down)):
            if n > 0:
                y = self.maxpool(xd[n - 1])
            else:
                y = x
            xd.append(self.down[n](y))
        return xd


class batchconvstyle(nn.Module):

    def __init__(self, in_channels, out_channels, style_channels, sz, conv_1D=False):
        super().__init__()
        self.concatenation = False
        self.conv = batchconv(in_channels, out_channels, sz, conv_1D)
        self.full = nn.Linear(style_channels, out_channels)

    def forward(self, style, x, mkldnn=False, y=None):
        if y is not None:
            x = x + y
        feat = self.full(style)
        for k in range(len(x.shape[2:])):
            feat = feat.unsqueeze(-1)
        if mkldnn:
            x = x.to_dense()
            y = (x + feat).to_mkldnn()
        else:
            y = x + feat
        y = self.conv(y)
        return y


class resup(nn.Module):

    def __init__(self, in_channels, out_channels, style_channels, sz, conv_1D=False):
        super().__init__()
        self.concatenation = False
        self.conv = nn.Sequential()
        self.conv.add_module("conv_0",
                             batchconv(in_channels, out_channels, sz, conv_1D=conv_1D))
        self.conv.add_module(
            "conv_1",
            batchconvstyle(out_channels, out_channels, style_channels, sz,
                           conv_1D=conv_1D))
        self.conv.add_module(
            "conv_2",
            batchconvstyle(out_channels, out_channels, style_channels, sz,
                           conv_1D=conv_1D))
        self.conv.add_module(
            "conv_3",
            batchconvstyle(out_channels, out_channels, style_channels, sz,
                           conv_1D=conv_1D))
        self.proj = batchconv0(in_channels, out_channels, 1, conv_1D=conv_1D)

    def forward(self, x, y, style, mkldnn=False):
        x = self.proj(x) + self.conv[1](style, self.conv[0](x), y=y, mkldnn=mkldnn)
        x = x + self.conv[3](style, self.conv[2](style, x, mkldnn=mkldnn),
                             mkldnn=mkldnn)
        return x


class make_style(nn.Module):

    def __init__(self, conv_1D=False):
        super().__init__()
        self.flatten = nn.Flatten()
        self.avg_pool = F.avg_pool1d if conv_1D else F.avg_pool2d

    def forward(self, x0):
        style = self.avg_pool(x0, kernel_size=x0.shape[2:])
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True)**.5
        return style


class upsample(nn.Module):

    def __init__(self, nbase, sz, conv_1D=False):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode="nearest")
        self.up = nn.Sequential()
        for n in range(1, len(nbase)):
            self.up.add_module("res_up_%d" % (n - 1),
                               resup(nbase[n], nbase[n - 1], nbase[-1], sz, conv_1D))

    def forward(self, style, xd, mkldnn=False):
        x = self.up[-1](xd[-1], xd[-1], style, mkldnn=mkldnn)
        for n in range(len(self.up) - 2, -1, -1):
            if mkldnn:
                x = self.upsampling(x.to_dense()).to_mkldnn()
            else:
                x = self.upsampling(x)
            x = self.up[n](x, xd[n], style, mkldnn=mkldnn)
        return x


