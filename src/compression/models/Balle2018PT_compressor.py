
import torch
from torch import nn
import math
import sys
sys.path.append('..')
from compression.models import abstract_model
from compression.models import GDN

class Balle2018PTTFExp_ImageCompressor(abstract_model.AbstractImageCompressor):
    '''
    This is the one
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320, lossf='mse', device='cuda:0', entropy_coding=False, **kwargs):
        super().__init__(out_channel_N=out_channel_N, out_channel_M=out_channel_M, lossf=lossf, device=device, entropy_coding=entropy_coding, **kwargs)
        self.Encoder = Analysis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.Decoder = Synthesis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorDecoder = SynthesisTFCodeExp_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorEncoder = Analysis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)

###################
# These are useless

class Balle2018PT_ImageCompressor(abstract_model.AbstractImageCompressor):
    def __init__(self, out_channel_N=192, out_channel_M=320, lossf='mse', device='cuda:0', entropy_coding=False, **kwargs):
        super().__init__(out_channel_N=out_channel_N, out_channel_M=out_channel_M, lossf=lossf, device=device, entropy_coding=entropy_coding, **kwargs)
        self.Encoder = Analysis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.Decoder = Synthesis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorDecoder = Synthesis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorEncoder = Analysis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)

class Balle2018PTStd_ImageCompressor(abstract_model.AbstractImageCompressor):
    def __init__(self, out_channel_N=192, out_channel_M=320, lossf='mse', device='cuda:0', entropy_coding=False, **kwargs):
        super().__init__(out_channel_N=out_channel_N, out_channel_M=out_channel_M, lossf=lossf, device=device, entropy_coding=entropy_coding, **kwargs)
        self.Encoder = Analysis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.Decoder = Synthesis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorDecoder = SynthesisStd_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorEncoder = Analysis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)

class Balle2018PTTF_ImageCompressor(abstract_model.AbstractImageCompressor):
    def __init__(self, out_channel_N=192, out_channel_M=320, lossf='mse', device='cuda:0', entropy_coding=False, **kwargs):
        super().__init__(out_channel_N=out_channel_N, out_channel_M=out_channel_M, lossf=lossf, device=device, entropy_coding=entropy_coding, **kwargs)
        self.Encoder = Analysis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.Decoder = Synthesis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorDecoder = SynthesisTFCode_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorEncoder = Analysis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)

class Balle2018PTPaperExp_ImageCompressor(abstract_model.AbstractImageCompressor):
    def __init__(self, out_channel_N=192, out_channel_M=320, lossf='mse', device='cuda:0', entropy_coding=False, **kwargs):
        super().__init__(out_channel_N=out_channel_N, out_channel_M=out_channel_M, lossf=lossf, device=device, entropy_coding=entropy_coding, **kwargs)
        self.Encoder = Analysis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.Decoder = Synthesis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorDecoder = SynthesisPaperExp_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorEncoder = Analysis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)

###################

class Analysis_net(nn.Module):
    '''
    Analysis net (liu's imprementation)
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320, activation_function='GDN'):
        super(Analysis_net, self).__init__()
        if activation_function == 'Hardswish':
            activation_function = torch.nn.Hardswish()
            self.gdn1 = activation_function
            self.gdn2 = activation_function
            self.gdn3 = activation_function
        elif activation_function == 'GDN':
            activation_function = GDN.GDN
            self.gdn1 = activation_function(out_channel_N)
            self.gdn2 = activation_function(out_channel_N)
            self.gdn3 = activation_function(out_channel_N)
        elif activation_function == 'GELU':
            activation_function = torch.nn.GELU
            self.gdn1 = activation_function()
            self.gdn2 = activation_function()
            self.gdn3 = activation_function()
        else:
            raise NotImplementedError(activation_function)

        self.conv1 = nn.Conv2d(3, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + out_channel_N) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)

        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)

        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)

        self.conv4 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn3(self.conv3(x))
        return self.conv4(x)


class Analysis_prior_net(nn.Module):
    '''
    Analysis prior net
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(Analysis_prior_net, self).__init__()
        self.conv1 = nn.Conv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.relu2 = nn.ReLU()
        self.conv3 =  nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)

    def forward(self, x):
        x = torch.abs(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.conv3(x)



class Synthesis_prior_net(nn.Module):
    '''
    Decode synthesis prior, liu's implementation
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(Synthesis_prior_net, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.relu1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.relu2 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_M, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, (math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        # self.priordecoder = nn.Sequential(
        #     nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(out_channel_N, out_channel_M, 3, stride=1, padding=1)
        # )

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        return torch.exp(self.deconv3(x))  # why exp ?? seems to increase runtime


class SynthesisPaperExp_prior_net(nn.Module):
    '''
    Decode synthesis prior, per Balle's paper
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.relu1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.relu2 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_M, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, (math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.relu3 = nn.ReLU()
        # self.priordecoder = nn.Sequential(
        #     nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(out_channel_N, out_channel_M, 3, stride=1, padding=1)
        # )

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        return torch.exp(self.relu3(self.deconv3(x)))

class SynthesisTFCode_prior_net(nn.Module):
    '''
    Decode synthesis prior, per Balle's code
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.relu1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.relu2 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_M, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, (math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        # self.priordecoder = nn.Sequential(
        #     nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(out_channel_N, out_channel_M, 3, stride=1, padding=1)
        # )

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        return self.deconv3(x)

class SynthesisTFCodeExp_prior_net(nn.Module):
    '''
    Decode synthesis prior, per Balle's code (with exp which may be present)
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.relu1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.relu2 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_M, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, (math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        # self.priordecoder = nn.Sequential(
        #     nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(out_channel_N, out_channel_M, 3, stride=1, padding=1)
        # )

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        return torch.exp(self.deconv3(x))


class SynthesisStd_prior_net(Synthesis_prior_net):
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super().__init__(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        return self.relu2(self.deconv3(x))

class Synthesis_net(nn.Module):
    '''
    Decode synthesis
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320, out_channel_fin=3, activation_function='GDN'):
        super(Synthesis_net, self).__init__()
        if activation_function == 'GDN':
            self.igdn1 = GDN.GDN(out_channel_N, inverse=True)
            self.igdn2 = GDN.GDN(out_channel_N, inverse=True)
            self.igdn3 = GDN.GDN(out_channel_N, inverse=True)
        elif activation_function == 'Hardswish':
            if activation_function == 'Hardswish':
                activation_function = torch.nn.Hardswish()
            self.igdn1 = activation_function
            self.igdn2 = activation_function
            self.igdn3 = activation_function
        else:
            if activation_function == 'GELU':
                activation_function = torch.nn.GELU
            else:
                raise NotImplementedError(activation_function)
            self.igdn1 = activation_function()
            self.igdn2 = activation_function()
            self.igdn3 = activation_function()
        self.deconv1 = nn.ConvTranspose2d(out_channel_M, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        #self.igdn1 = GDN.GDN(out_channel_N, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        #self.igdn2 = GDN.GDN(out_channel_N, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        #self.igdn3 = GDN.GDN(out_channel_N, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(out_channel_N, out_channel_fin, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, (math.sqrt(2 * 1 * (out_channel_N + 3) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)


    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.igdn3(self.deconv3(x))
        x = self.deconv4(x)
        return x

class Analysis2017_net(nn.Module):
    '''
    Analysis net
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(Analysis2017_net, self).__init__()
        self.conv1 = nn.Conv2d(3, out_channel_N, 9, stride=4, padding=4)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + out_channel_N) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN.GDN(out_channel_N)
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN.GDN(out_channel_N)
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN.GDN(out_channel_M)


    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn3(self.conv3(x))
        return x

class Synthesis2017_net(nn.Module):
    '''
    Decode synthesis
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320, out_channel_fin=3):
        super(Synthesis2017_net, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_M, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn1 = GDN.GDN(out_channel_M, inverse=True)

        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn2 = GDN.GDN(out_channel_N, inverse=True)

        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_fin, 9, stride=4, padding=3, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, (math.sqrt(2 * 1 * (out_channel_N + 3) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.igdn3 = GDN.GDN(out_channel_N, inverse=True)


    def forward(self, x):
        x = self.deconv1(self.igdn1(x))
        x = self.deconv2(self.igdn2(x))
        x = self.deconv3(self.igdn3(x))
        return x