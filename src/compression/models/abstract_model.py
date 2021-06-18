# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
import math
import time
#import thop
import ptflops
import numpy as np
import logging
import sys
import statistics
sys.path.append('..')
from compression.models.bitEstimator import BitEstimator
from common.extlibs import pt_ms_ssim
from common.libs import pt_ops
from compression.models import GDN
try:
    import torchac
except ModuleNotFoundError:
    print('abstract_model: torchac not available; entropy coding is disabled')

logger = logging.getLogger("ImageCompression")

GAUSSIAN_DISTRIBUTION_USES_LUT = False
NUMSCALES = 64 # 384 is the max I can fit with 48GB of RAM

def _renorm_cast_cdf_(cdf, precision):
    Lp = cdf.shape[-1]
    finals = 1  # NHW1
    # RENORMALIZATION_FACTOR in cuda
    f = torch.tensor(2, dtype=torch.float32, device=cdf.device).pow_(precision)
    cdf = cdf.mul((f - (Lp -1)) / finals)  # TODO # DBG, was -1
    #breakpoint()
    r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
    #r = torch.ones_like(cdf)
    cdf.add_(r)
    #breakpoint()
    cdf = cdf.round()

    cdf = cdf.to(dtype=torch.int16, non_blocking=False)
    return cdf

class AbstractImageCompressor(nn.Module):
    def __init__(self, out_channel_N=192, out_channel_M=320, lossf='mse', device='cuda:0', min_feat=-127, max_feat=128, min_feat_gaussian=-127, max_feat_gaussian=128, q_intv=1, precision=16, entropy_coding=False, conditional_distribution='Laplace', passthrough_ae=False, **kwargs):#, out_channel_M=320, lossf='mse'):
        super(AbstractImageCompressor, self).__init__()
        # DBG: targets should be 0.5?
        self.entropy_coding = entropy_coding and 'torchac' in sys.modules
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M
        self.bitEstimator_z = BitEstimator(out_channel_N)
        #self.out_channel_N = out_channel_N
        #self.out_channel_M = out_channel_M
        self.lossf = lossf
        if isinstance(lossf, str):
            if lossf == 'mse':
                self.lossfun = F.mse_loss
            elif lossf == 'ssim':
                self.lossclass = pt_ms_ssim.SSIM()
                self.lossfun = self.lossclass.lossfun
            elif lossf == 'msssim':
                self.lossclass = pt_ms_ssim.MS_SSIM()
                self.lossfun = self.lossclass.lossfun
            else:
                raise ValueError(lossf)
        else:
            self.lossfun = lossf
        self.device = device
        self.min_feat, self.max_feat, self.q_intv, self.precision = min_feat, max_feat, q_intv, precision
        self.min_feat_gaussian, self.max_feat_gaussian = min_feat_gaussian, max_feat_gaussian
        ntargets = int((-self.min_feat+self.max_feat)/self.q_intv+1)
        #self.fixed_entropy_table = torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(out_channel_M, ntargets)))
        self.z_entropy_table = torch.empty(out_channel_M, ntargets, dtype=torch.short)
        if conditional_distribution == 'Laplace':
            self.conditional_distribution = torch.distributions.laplace.Laplace
        elif conditional_distribution == 'Gaussian' or conditional_distribution == 'Normal':
            self.conditional_distribution = torch.distributions.normal.Normal
        self.num_distributions = 1
        self.frozen_autoencoder = False
        self.passthrough_ae = passthrough_ae
        #if passthrough_ae:
        #    self.roundNoGradient = pt_ops.RoundNoGradient()
        # DBG
        # self.min_feat = -255
        # self.max_feat = 256

        #self.visual_loss =

    def freeze_autoencoder(self):
        for param in self.Encoder.parameters():
            param.requires_grad = False
        for param in self.Decoder.parameters():
            param.requires_grad = False
        self.bak_lossfun = self.lossfun
        self.lossfun = pt_ops.oneloss
        self.frozen_autoencoder = True

    def unfreeze_autoencoder(self):
        for param in self.Encoder.parameters():
            param.requires_grad = True
        for param in self.Decoder.parameters():
            param.requires_grad = True
        self.lossfun = self.bak_lossfun
        del(self.bak_lossfun)
        self.frozen_autoencoder = False


    def forward(self, input_image):
        quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_M, input_image.size(2) // 16, input_image.size(3) // 16, device=self.device)
        quant_noise_z = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 64, input_image.size(3) // 64, device=self.device)
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5, 0.5)
        feature = self.Encoder(input_image)
        batch_size = feature.size()[0]
        z = self.priorEncoder(feature)

        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)
        recon_sigma = self.priorDecoder(compressed_z)


        if self.training:
            quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_M, input_image.size(2) // 16, input_image.size(3) // 16, device=self.device)
            quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
            compressed_feature_entropy = feature + quant_noise_feature
            if self.passthrough_ae:  # ae: round, entropy: noise

                compressed_feature_ae = pt_ops.RoundNoGradient().apply(feature)
            else:  # ae: noise, entropy: noise
                compressed_feature_ae = compressed_feature_entropy
        else:
            # ae and entropy: round
            compressed_feature_entropy = torch.round(feature)
            compressed_feature_ae = compressed_feature_entropy

        recon_image = self.Decoder(compressed_feature_ae)
        clipped_recon_image = recon_image.clamp(0., 1.)

        visual_loss = self.lossfun(recon_image, input_image)

        def feature_probs_based_sigma(feature, sigma):
            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-10, 1e10)
            gaussian = self.conditional_distribution(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, probs

        def iclr18_estimate_bits_z(z):
            #z = z[0, :, 2, 2].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            #breakpoint()
            return total_bits, prob
        total_bits_feature, _ = feature_probs_based_sigma(compressed_feature_entropy, recon_sigma)

        im_shape = input_image.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        if self.training or not self.entropy_coding:
            total_bits_z, _ = iclr18_estimate_bits_z(compressed_z)
        else:  # eval
            print('estimate: {}'.format(iclr18_estimate_bits_z(compressed_z)[0])) # dbg
            bitstream_z, numbytes_z = self.entropy_encode(compressed_z)
            total_bits_z = numbytes_z * 8
            print('actual bits z: {}'.format(total_bits_z))
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z
        #tbl = self.build_table_z(z, self.bitEstimator_z)
        #breakpoint()
        return clipped_recon_image, visual_loss, bpp_feature, bpp_z, bpp, None, None

    def update_entropy_table(self):
        # side string learned entropy coding
        # # https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html
        ntargets = int((-self.min_feat+self.max_feat)/self.q_intv+2)  # +2 = 19000->6000 DBG
        targets = torch.linspace(self.min_feat-.5, self.max_feat+.5, ntargets)
        targets = targets.expand(self.out_channel_N, ntargets).unsqueeze(0).unsqueeze(-1)  # 1, nch, L, 1
        probs = self.bitEstimator_z(targets.to(self.device))

        self.z_entropy_table = _renorm_cast_cdf_(#DBG
            (probs).squeeze(-1).unsqueeze(0), self.precision).squeeze(0).squeeze(0) # 1,1,ch,L -> ch,L
        self.z_entropy_table_float = self.entropy_table_float = probs.squeeze()

        # main features: gaussian conditional entropy
        ntargets = int((-self.min_feat_gaussian+self.max_feat_gaussian)/self.q_intv+2)  # +2 = 19000->6000 DBG
        targets = torch.linspace(self.min_feat_gaussian-.5, self.max_feat_gaussian+.5, ntargets)
        # #breakpoint()
        self.main_targets = targets

        self.scale_table = np.exp(np.linspace(np.log(0.11), np.log(256), NUMSCALES))#256
        self.scale_table = np.exp(np.linspace(np.log(0.0001), np.log(128), NUMSCALES))#256
        self.conditional_CDF_float = torch.zeros(NUMSCALES, ntargets)
        self.conditional_CDF = torch.zeros(NUMSCALES, ntargets).short()

        probs = torch.zeros(NUMSCALES, ntargets)
        for scalenum, ascale in enumerate(self.scale_table):
            self.conditional_CDF_float[scalenum] = self.conditional_distribution(0, ascale).cdf(self.main_targets)
            self.conditional_CDF[scalenum] = _renorm_cast_cdf_(self.conditional_CDF_float[scalenum].unsqueeze(0), self.precision).squeeze(0).squeeze(0)
        # self.conditional_CDF = _renorm_cast_cdf_(#DBG
        #     (probs).squeeze(-1).unsqueeze(0), self.precision).squeeze(0).squeeze(0)


    def eval(self):
        if self.entropy_coding:
            self.update_entropy_table()
        return self.train(False)

    def entropy_encode(self, z):
        '''entropy encode a whole sidestring'''
        #return self.entropy_encode_per_channel(z)#DBG
        bitstream = []
        nbytes = 0
        bs, nch, h, w = z.shape
        nL = self.z_entropy_table.size(-1)
        for bn in range(bs):
            cdf_tbl = self.z_entropy_table.expand(1, h*w, nch, nL).contiguous()
            #breakpoint()
            z_int = z[bn].reshape(1,nch,-1).transpose(1,2).contiguous()
            z_int = ((z_int - self.min_feat) / self.q_intv).round().short().flatten()
            encoded_str = torchac.encode_cdf(cdf_tbl, z_int)
            bitstream.append(encoded_str)
            nbytes += len(encoded_str)
            decoded_str = torchac.decode_cdf(cdf_tbl, encoded_str)
            #print(encoded_str)
            #print(z_int)
            #print(decoded_str)
            decoded_str = (decoded_str.view(1,h*w,nch).transpose(1,2).reshape(1,nch,h,w)*self.q_intv+self.min_feat)
            if not torch.equal(z, decoded_str.float()):
                logger.info('entropy_encode error: z and decoded_str do not match.')
                logger.info(cdf_tbl)
                breakpoint()
            elif z.max() > self.max_feat or z.min() < self.min_feat:
                logger.info('entropy_encode error: z out of range: [] instead of []').format(z.min(), z.max(), self.min_feat, self.max_feat)
                breakpoint()
                #assert torch.equal(z_int, decoded_str), (z_int, decoded_str)
        logger.info(nbytes*8)
        #breakpoint()
        return bitstream, nbytes

    def encode(self, input_image, entropy_coding=True):
        '''
        egtest:
            python train.py --test_flags testKodakTiming --arch Balle2018PT --pretrain mse_4096_swopt
            TODO save to file (like manynets implementation)

        '''
        bs, in_ch, in_h, in_w = input_image.shape
        feature = self.Encoder(input_image)
        feature_renorm = torch.round(feature)
        feature_renorm = ((feature_renorm.flatten()-self.min_feat_gaussian) / self.q_intv).round().short()
        z = torch.round(self.priorEncoder(feature))
        bs, z_nch, z_h, z_w = z.shape
        z_nL = self.z_entropy_table.size(-1)
        cdf_z = self.z_entropy_table.unsqueeze(0).expand(z_h*z_w, z_nch, z_nL).unsqueeze(0)
        z_int = ((z.view(1, z_nch, -1).transpose(1,2)-self.min_feat) / self.q_intv).round().short().flatten()
        # if entropy_coding:
        #     encoded_zstr = torchac.encode_cdf(cdf_tbl, z_int)  # comment to rm entropy coding

        recon_sigma = self.priorDecoder(z)

        mu = torch.zeros_like(recon_sigma).flatten()  # loc
        sigma = recon_sigma.clamp(1e-10, 1e10).flatten()  # scale
        if GAUSSIAN_DISTRIBUTION_USES_LUT:
            pass
        else:
            gaussian = self.conditional_distribution(mu.flatten(), sigma.flatten())
            cdf_main = gaussian.cdf(self.main_targets.unsqueeze(-1).expand(-1, sigma.size(0))).transpose(0,1).unsqueeze(0).unsqueeze(0)
            cdf_main = _renorm_cast_cdf_(cdf_main, self.precision)
        #cdfmaintest = cdf_main.cpu()[:,:,0:100,:].contiguous()
        #feature_renormtest = feature_renorm[0:100].contiguous()
        #breakpoint()
        #astrtest = torchac.encode_cdf(cdfmaintest, feature_renormtest)
        if not entropy_coding:
            return feature_renorm, z_int
        encoded_mstr = torchac.encode_cdf(cdf_main.cpu(), feature_renorm.cpu())

        #breakpoint()
        #probs_main = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
        #z2enc = z.view(z_nch,-1).transpose(0,1).unsqueeze(-1).contiguous().long()
        #z_indices_minus = (-self.min_feat+z2enc).unsqueeze(0)
        #z_indices_plus = (self.max_feat+z2enc).unsqueeze(0)
        #probs_z = torch.gather(cdf_tbl, 2, z_indices_plus) - torch.gather(cdf_tbl, 2, z_indices_minus)
        encoded_zstr = torchac.encode_cdf(cdf_z.cpu(), z_int.cpu())

        ''' TODO
        Do I need to encode the CDF for every possible value L at each ch*h*w location ?
        '''
        #breakpoint()
        #all_probs_main = gaussian.cdf(self.main_targets)
        #entropy_table_main = _renorm_cast_cdf_(all_probs_main.squeeze(-1).unsqueeze(0), self.precision).squeeze(0).squeeze(0)
        #breakpoint()
        #cdf_tbl_main = entropy_table_main.unsqueeze(0).expand(feat_h*feat_w, self.out_channel_M, self.main_targets.size(-2)).unsqueeze(0)
        #encoded_main_str = torchac.encode_cdf(cdf_tbl_main.cpu(), feature_renorm.cpu())
        # means = mu
        # log_scales = sigma
        # sym = feature_renorm
        # logit_probs_softmax = probs_main
        # targets = None


        # torchac.encode_logistic_mixture(targets, means, log_scales, logit_probs_softmax, sym)
        #breakpoint()
        ### entropy-encode main string

        ###

        return [encoded_mstr, encoded_zstr, in_h, in_w]

    def decode(self, bitstream):
        if isinstance(bitstream, torch.Tensor):
            # for testing complexity
            HARDCODED_ZCH = 192
            zshape = [bitstream.size(0), HARDCODED_ZCH, bitstream.size(2)//4, bitstream.size(3)//4]
            z = torch.rand(zshape)
            z = self.priorDecoder(z)
            decoded_mstr = self.Decoder(bitstream)
        else:
            main_string, zstring, img_h, img_w = bitstream
            z_nL = self.z_entropy_table.size(-1)
            feat_h = img_h // 16
            feat_w = img_w // 16
            z_h = feat_h // 16
            z_w = feat_w // 16
            z_nch = self.out_channel_N
            cdf_z = self.z_entropy_table.unsqueeze(0).expand(z_h*z_w, z_nch, z_nL).unsqueeze(0)
            z = torchac.decode_cdf(cdf_z.cpu(), zstring).to(self.device)
            z = z * self.q_intv + self.min_feat
            breakpoint()
            z = z.view(1, -1, z_nch).transpose(1,2).view(1, z_nch, z_h, z_w).float()
            recon_sigma = self.priorDecoder(z)
            mu = torch.zeros_like(recon_sigma).flatten()
            sigma = recon_sigma.clamp(1e-10, 1e10).flatten()
            gaussian = self.conditional_distribution(mu.flatten(), sigma.flatten())
            cdf_main = gaussian.cdf(self.main_targets.unsqueeze(-1).expand(-1, sigma.size(0))).transpose(0,1).unsqueeze(0).unsqueeze(0)
            cdf_main = _renorm_cast_cdf_(cdf_main, self.precision)
            decoded_mstr = torchac.decode_cdf(cdf_main.cpu(), main_string)
            decoded_mstr = decoded_mstr * self.q_intv + self.min_feat
            decoded_mstr = decoded_mstr.view(1, 3, feat_h, feat_w)
        return decoded_mstr

    def timing_analysis(self, input_image, entropy_coding=False):
        '''
        egrun
        python train.py --pretrain mse_4096_tfcodeexp_adam --test_flags timing --device -1 --arch Balle2018PTTFExp
        '''
        self.update_entropy_table()
        timedict = {'nnenc_main': [], 'nnenc_hp': [], 'nndec_hp': [], 'nndec_main': [], 'make_cdf_tbl': [],  'make_cdf_tbl_lut': [], 'torchac_enc_main': [], 'torchac_enc_hp': [], 'torchac_dec_hp': [], 'torchac_dec_main': []}

        for rep in range(50):
            with torch.no_grad():
                tic = time.perf_counter()
                bs, in_ch, in_h, in_w = input_image.shape
                feature = self.Encoder(input_image)
                feature_renorm = torch.round(feature)
                timedict['nnenc_main'].append(time.perf_counter()-tic)

                tic = time.perf_counter()
                z = torch.round(self.priorEncoder(feature))
                timedict['nnenc_hp'].append(time.perf_counter()-tic)




                tic = time.perf_counter()
                bs, z_nch, z_h, z_w = z.shape
                z_nL = self.z_entropy_table.size(-1)
                cdf_z = self.z_entropy_table.unsqueeze(0).expand(z_h*z_w, z_nch, z_nL).unsqueeze(0)
                z_int = ((z.view(1, z_nch, -1).transpose(1,2)-self.min_feat) / self.q_intv).round().short().flatten()
                encoded_zstr = torchac.encode_cdf(cdf_z.contiguous(), z_int.contiguous())  # comment to rm entropy coding
                timedict['torchac_enc_hp'].append(time.perf_counter()-tic)

                tic = time.perf_counter()
                decoded_zstr = torchac.decode_cdf(cdf_z.contiguous(), encoded_zstr)
                timedict['torchac_dec_hp'].append(time.perf_counter()-tic)

                tic = time.perf_counter()
                recon_sigma = self.priorDecoder(z)
                timedict['nndec_hp'].append(time.perf_counter()-tic)

                tic = time.perf_counter()
                mu = torch.zeros_like(recon_sigma).flatten()  # loc
                sigma = recon_sigma.clamp(1e-10, 1e10).flatten()  # scale

                if GAUSSIAN_DISTRIBUTION_USES_LUT:
                    raise NotImplementedError
                else:
                    gaussian = self.conditional_distribution(mu.flatten(), sigma.flatten())
                    cdf_main = gaussian.cdf(self.main_targets.unsqueeze(-1).expand(-1, sigma.size(0))).transpose(0,1).unsqueeze(0).unsqueeze(0)
                    cdf_main = _renorm_cast_cdf_(cdf_main, self.precision)
                timedict['make_cdf_tbl'].append(time.perf_counter()-tic)
                #cdfmaintest = cdf_main.cpu()[:,:,0:100,:].contiguous()
                #feature_renormtest = feature_renorm[0:100].contiguous()
                #breakpoint()
                #astrtest = torchac.encode_cdf(cdfmaintest, feature_renormtest)

                tic = time.perf_counter()
                #if not entropy_coding:
                #    return feature_renorm, z_int
                feature_renorm_short = ((feature_renorm.flatten()-self.min_feat_gaussian) / self.q_intv).round().short()
                encoded_mstr = torchac.encode_cdf(cdf_main, feature_renorm_short)
                timedict['torchac_enc_main'].append(time.perf_counter()-tic)


                tic = time.perf_counter()
                decoded_features = self.Decoder(feature_renorm)
                timedict['nndec_main'].append(time.perf_counter()-tic)

                tic = time.perf_counter()
                decoded_mstr = torchac.decode_cdf(cdf_main, encoded_mstr)
                timedict['torchac_dec_main'].append(time.perf_counter()-tic)

                tic = time.perf_counter()
                _, feat_nch, feat_h, feat_w = feature.shape
                cdf_indices = (recon_sigma-torch.tensor(self.scale_table).view(NUMSCALES,1,1,1).expand(NUMSCALES,feat_nch,feat_h,feat_w)).abs().argmin(0)
                exp_cdf = self.conditional_CDF.view(-1, 1, 1, z_nL) # 64,1,1,257)
                exp_cdf = exp_cdf.expand(-1, feat_nch, feat_h*feat_w, z_nL)
                cdf_indices = cdf_indices.view(1,feat_nch, -1,1).expand(-1,-1,-1,z_nL)
                cdf_tbl_main = torch.gather(exp_cdf, 0, cdf_indices)
                timedict['make_cdf_tbl_lut'].append(time.perf_counter()-tic)

                logger.info(timedict)

                # check accuracy w/ quantization

                exp_cdf_flt = self.conditional_CDF_float.view(-1, 1, 1, z_nL) # 64,1,1,257)
                exp_cdf_flt = exp_cdf_flt.expand(-1, feat_nch, feat_h*feat_w, z_nL)
                cdf_tbl_main_flt = torch.gather(exp_cdf_flt, 0, cdf_indices)
                feat_indices_lower = (feature_renorm.view(1,feat_nch,-1,1)).long()-self.min_feat
                feat_indices_upper = feat_indices_lower + 1
                probs_flt = (torch.gather(cdf_tbl_main_flt, 3, feat_indices_upper) - torch.gather(cdf_tbl_main_flt, 3, feat_indices_lower)).float()
                #cdf_indices[:,:,:,0].reshape(1,256,-1)
                #exp_cdf[:,:,:,0]
                #probs_flt = self.conditional_CDF_float.view(-1, 1, 1, z_nL) # 64,1,1,257)
                #probs_flt = probs_flt.expand(-1, feat_nch, feat_h*feat_w, z_nL)
                #recon_sigma_quant = torch.gather(exp_cdf_flt[:,:,:,0], 0, cdf_indices[:,:,:,0].reshape(1,256,-1)).view(recon_sigma.shape)
                #g#aussian_quant = self.conditional_distribution(mu.view(recon_sigma_quant.shape).contiguous(), recon_sigma_quant.contiguous())
                #probs = gaussian_quant.cdf(feature + 0.5) - gaussian_quant.cdf(feature - 0.5)
                #breakpoint()
                total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs_flt + 1e-10) / math.log(2.0), 0, 50))
                logger.info('quant: ')
                logger.info(total_bits)
                sigma = recon_sigma.clamp(1e-10, 1e10)
                gaussian = self.conditional_distribution(mu.view(sigma.shape).contiguous(), sigma)
                probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
                total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
                logger.info('noquant: ')
                logger.info(total_bits)
                #breakpoint()



        for ares in timedict:
            timedict[ares] = statistics.mean(timedict[ares])
        logger.info('\n')
        logger.info(timedict)
        return timedict


    def entropy_encode_per_channel(self, z):
        '''entropy encode a sidestring per channel (useful for debugging)'''
        bitstream = []
        nbytes = 0
        bs, nch, h, w = z.shape
        #z -= 0.5
        for bn in range(bs):
            for ch in range(nch):
                cdf_tbl = self.z_entropy_table[ch].expand(1, h, w, self.z_entropy_table.size(-1)).contiguous()
                z_int = ((z[bn, ch] - self.min_feat) / self.q_intv).round().short().flatten()
                encoded_str = torchac.encode_cdf(cdf_tbl, z_int)
                bitstream.append(encoded_str)
                nbytes += len(encoded_str)
                decoded_str = torchac.decode_cdf(cdf_tbl, encoded_str)
                logger.info(ch)
                logger.info('encoded_str')
                logger.info(encoded_str)
                #print(z_int)

                if not torch.equal(z_int, decoded_str) or ch == 5 or ch == 6:  # DBG
                    logger.info(decoded_str)
                    logger.info('z, z_int, z_entropy_table')
                    logger.info(z[bn][ch].flatten()) #dbg
                    logger.info(z_int)
                    logger.info(self.z_entropy_table[ch]) #dbg
                    breakpoint()
                #assert torch.equal(z_int, decoded_str), (z_int, decoded_str)
        logger.info(nbytes*8)
        breakpoint()
        return bitstream, nbytes



    def get_parameters(self, lr=None):
        assert lr is not None
        param_list = [
                {'params': self.Encoder.parameters(), 'name': 'encoder'},
                {'params': self.Decoder.parameters(), 'name': 'decoder'},
                {'params': self.priorEncoder.parameters(), 'name': 'prior_encoder'},
                {'params': self.priorDecoder.parameters(), 'name': 'prior_decoder'},
                {'params': self.bitEstimator_z.parameters(), 'lr': lr*10, 'name': 'bit_estimator'},
                ]  # Note that bit_estimator must be last
        return param_list
        #return self.parameters()

    def compress(self, input_image):
        feature = self.Encoder(input_image)
        z = self.priorEncoder(feature)
        compressed_z = torch.round(z)
        compressed_feature = torch.round(feature)
        return compressed_feature, compressed_z

    def get_encoding_class(self):
        return Encoder(self)

    def get_decoding_class(self):
        return Decoder(self)

    def complexity_analysis(self):
        '''
        egrun:
            CUDA_AVAILABLE_DEVICES="" python train.py --test_flags complexity --pretrain mse_4096_b2017manypriors_64pr_16px_adam_2upd_d --device -1

        '''
        IMGCH, IMGHEIGHT, IMGWIDTH = 3, 2160, 3840
        LATENTCH = 256
        # TODO add custom GDN hook
        encoder = self.get_encoding_class()
        #res = encoder(test_img)
        #breakpoint()
        #print(test_img.shape)
        macs, params = ptflops.get_model_complexity_info(encoder, (IMGCH, IMGHEIGHT, IMGWIDTH), custom_modules_hooks = {GDN.GDN: GDN.gdn_flops_counter_hook})

        decoder = self.get_decoding_class()
        macs, params = ptflops.get_model_complexity_info(decoder, (LATENTCH, IMGHEIGHT//16, IMGWIDTH//16), custom_modules_hooks = {GDN.GDN: GDN.gdn_flops_counter_hook})

    def cpu(self):
        self.device = torch.device('cpu')
        return self.to(self.device)


class Encoder(nn.Module):
    def __init__(self, compressor):
        super().__init__()
        self.compressor = compressor
    def forward(self, input_image):
        return self.compressor.encode(input_image, entropy_coding=False)

class Decoder(nn.Module):
    def __init__(self, compressor):
        super().__init__()
        self.compressor = compressor
    def forward(self, input_image):
        return self.compressor.decode(input_image)

