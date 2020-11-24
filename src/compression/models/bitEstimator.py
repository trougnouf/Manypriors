# from .basics import *
# import pickle
# import os
# import codecs
import torch.nn as nn
import torch
import torch.nn.functional as F

class Bitparm(nn.Module):
    '''
    save params
    '''
    def __init__(self, channel, final=False):
        super(Bitparm, self).__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)

class BitEstimator(nn.Module):
    '''
    Estimate bit
    '''
    def __init__(self, channel, **kwargs):
        super(BitEstimator, self).__init__()
        self.f1 = Bitparm(channel)
        self.f2 = Bitparm(channel)
        self.f3 = Bitparm(channel)
        self.f4 = Bitparm(channel, True)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)

class BitparmSingle(nn.Module):
    '''
    save params
    '''
    def __init__(self, channel, final=False):
        super(BitparmSingle, self).__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)

class BitEstimatorSingle(nn.Module):
    '''
    Estimate bit
    from a single dimension array
    '''
    def __init__(self, channel, **kwargs):
        super(BitEstimatorSingle, self).__init__()
        self.f1 = BitparmSingle(channel)
        self.f2 = BitparmSingle(channel)
        self.f3 = BitparmSingle(channel)
        self.f4 = BitparmSingle(channel, True)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)


class MultiHeadBitEstimator(nn.Module):
    '''
    Estimate bit
    '''
    def __init__(self, channel: int, nb_head: int, shape=('g', 'bs', 'ch', 'h', 'w'), bitparm_init_mode='normal', bitparm_init_range=0.01, **kwargs):
        super(MultiHeadBitEstimator, self).__init__()
        self.f1 = MultiHeadBitparm(channel, nb_head=nb_head, shape=shape, bitparm_init_mode=bitparm_init_mode, bitparm_init_range=bitparm_init_range)
        self.f2 = MultiHeadBitparm(channel, nb_head=nb_head, shape=shape, bitparm_init_mode=bitparm_init_mode, bitparm_init_range=bitparm_init_range)
        self.f3 = MultiHeadBitparm(channel, nb_head=nb_head, shape=shape, bitparm_init_mode=bitparm_init_mode, bitparm_init_range=bitparm_init_range)
        self.f4 = MultiHeadBitparm(channel, final=True, nb_head=nb_head, shape=shape, bitparm_init_mode=bitparm_init_mode, bitparm_init_range=bitparm_init_range)
#        if bs_first:
#            self.prep_input_fun = lambda x: x.unsqueeze(0)
#        else:
#            self.prep_input_fun = lambda x: x

    def forward(self, x):
        #x = self.prep_input_fun(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)

class MultiHeadBitparm(nn.Module):
    '''
    save params
    '''
    def __init__(self, channel, nb_head, final=False, shape=('g', 'bs', 'ch', 'h', 'w'), bitparm_init_mode='normal', bitparm_init_range=0.01):
        super(MultiHeadBitparm, self).__init__()
        self.final = final
        if shape == ('g', 'bs', 'ch', 'h', 'w'):  # used in Balle2017ManyPriors_ImageCompressor
            params_shape = (nb_head, 1, channel, 1, 1)
        elif shape == ('bs','ch','g','h','w'):
            params_shape = (1, channel, nb_head, 1, 1)
        #shape = (nb_head, 1, channel, 1, 1) if bs_first else (1, nb_head, channel, 1, 1)
        if bitparm_init_mode == 'normal':
            init_fun = torch.nn.init.normal_
            init_params = 0, bitparm_init_range
        elif bitparm_init_mode == 'xavier_uniform':
            init_fun = torch.nn.init.xavier_uniform_
            init_params = [bitparm_init_range]
        else:
            raise NotImplementedError(bitparm_init_mode)
        self.h = nn.Parameter(init_fun(torch.empty(nb_head, channel).view(params_shape), *init_params))
        self.b = nn.Parameter(init_fun(torch.empty(nb_head, channel).view(params_shape), *init_params))
        if not final:
            self.a = nn.Parameter(init_fun(torch.empty(nb_head, channel).view(params_shape), *init_params))
        else:
            self.a = None

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)



if __name__ == '__main__':
    # doodling around w/ lr opt and size and such
    import sys
    sys.path.append('..')
    from liujiaheng_compression.models import bitEstimator
    from common.extlibs import radam
    import torch
    import math
    import numpy as np
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ane = bitEstimator.BitEstimator(256).cuda()
    bat = torch.rand((4,256,16,16)).cuda()*10
    bs=1
    bat = torch.rand((bs,256,2,2)).cuda()*50
    #bat = torch.ones((1,256,1,1), requires_grad=True)

    #bat.require_grad = True
    optimizer = radam.RAdam(ane.parameters(), lr=0.05)
    for i in range(100):
        optimizer.zero_grad()
        bast_quant = bat + torch.rand_like(bat)/2 - torch.rand_like(bat)/2
        probs = ane(bast_quant+0.5) - ane(bast_quant-0.5)
        loss=torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 10))/(32*32*bs)
        loss.backward()
        optimizer.step()
        print(loss)
        #bat = bat*(torch.rand_like(bat))
        bat += (torch.rand_like(bat)/10)
        bat -= (torch.rand_like(bat)/10)
'''
2000 steps
# @/10000 w/adam
#lr=1 0.0921
#lr=0.5 0.0196
#lr=0.1 0.0064
#lr=0.05 0.0052
#lr=0.01 0.0106
#lr=0.001 0.1597
#lr=0.0001 2.6511
radam @/10000
1 0.1475
0.1 0.0062
0.05 0.0064
0.03 .0089
lr=0.01 0.0246
radam @/10000 5000st
0.05 0.0066
0.03 0.0064
radam @/1000 5000st
0.01 0.0750
radam @/10 5000st
0.01 3.579
0.001 3.5851
0.0001 3.6350
bs2 d2 *10 /2 10000s
0.1
'''