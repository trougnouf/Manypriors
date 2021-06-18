
import torch
from torch import nn
import math
import random
import numpy as np
import time
import statistics
import logging
import sys
sys.path.append('..')
from compression.models import abstract_model
from compression.models import Balle2018PT_compressor
from compression.models import bitEstimator
from common.libs import pt_helpers
from common.libs import pt_ops
from common.libs import distinct_colors
from common.libs import utilities
logger = logging.getLogger("ImageCompression")
try:
    import torchac
except ModuleNotFoundError:
    logger.info('manynets_compressor: torchac not available; entropy coding is disabled')
try:
    import png
except ModuleNotFoundError:
    logger.info('manynets_compressor: png is not available (currently used in encode/decode)')


class Balle2017ManyPriors_ImageCompressor(abstract_model.AbstractImageCompressor):
    def __init__(self, out_channel_N=192, out_channel_M=320, lossf='mse', device='cuda:0', num_distributions = 64, dist_patch_size=1, nchans_per_prior=None, min_feat=-127, max_feat=128, q_intv=1, precision=16, entropy_coding=False, model_param='2018', activation_function='GDN', passthrough_ae=False, encoder_cls=None, decoder_cls=None, **kwargs):
        '''
        max cost to encode the prior: (bits_per_prior) / (patch_size)**2; typically 6 * (16**2) = 0.0234375
        for SpaceChans encoding:
        (nch * (bits_per_prior)) / (patch_size**2 * nchan_per_prior) bpp for priors
        eg: (256*8)/(128**2 * 4) = 0.0313 bpp
        eg: (256*8)/(32**2 * 64) = 0.0313

        model_param:
            "2017" for original Balle2017 paper
            "2018" to use encoder/decoder subnetwork of Balle2018 paper (default)
        '''
        super().__init__(out_channel_N=out_channel_N, out_channel_M=out_channel_M, lossf=lossf, device=device, min_feat=min_feat, max_feat=max_feat, q_intv=q_intv, precision=precision, entropy_coding=entropy_coding, passthrough_ae=passthrough_ae)
        self.nchans_per_prior = nchans_per_prior
        if self.nchans_per_prior is None:
            self.nchans_per_prior = out_channel_M
        del self.bitEstimator_z
        if encoder_cls is not None and decoder_cls is not None:
            self.Encoder = encoder_cls(out_channel_N=out_channel_N, out_channel_M=out_channel_M, activation_function=activation_function)
            self.Decoder = decoder_cls(out_channel_N=out_channel_N, out_channel_M=out_channel_M, activation_function=activation_function)
        if model_param == '2018' or model_param is None or model_param == 'None':
            self.Encoder = Balle2018PT_compressor.Analysis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M, activation_function=activation_function)
            self.Decoder = Balle2018PT_compressor.Synthesis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M, activation_function=activation_function)
        elif model_param == '2017':
            self.Encoder = Balle2018PT_compressor.Analysis2017_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
            self.Decoder = Balle2018PT_compressor.Synthesis2017_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        else:
            raise NotImplementedError(f"model_param={model_param}")
        #self.bitEstimators = nn.ModuleList()
        self.num_distributions = num_distributions
        self.dist_patch_size = dist_patch_size
        self.bitEstimators = bitEstimator.MultiHeadBitEstimator(self.nchans_per_prior, nb_head=self.num_distributions, shape=('g', 'bs', 'ch', 'h', 'w'), **kwargs)
        #self.previously_unused_dists = np.arange(self.num_distributions)
        self.dists_last_use = np.zeros(self.num_distributions, dtype=int)
        ntargets = int((-self.min_feat+self.max_feat)/self.q_intv+1)
        self.entropy_table = torch.zeros(self.num_distributions, out_channel_M, ntargets, dtype=torch.short)

        # for i in range(self.num_distributions):
        #     self.bitEstimators.append(bitEstimator.BitEstimator(out_channel_M))

    def update_entropy_table(self):
        try:
            ntargets = int((-self.min_feat+self.max_feat)/self.q_intv+2)
            targets = torch.linspace(self.min_feat-.5, self.max_feat+.5, ntargets).to(self.device)
            targets = targets.expand(1, self.out_channel_M, ntargets).unsqueeze(-1).contiguous()  # contiguous?
            entropy_table = self.bitEstimators(targets).squeeze(-1).squeeze(1)
            self.entropy_table_float = entropy_table.squeeze()
            self.entropy_table = torchac._renorm_cast_cdf_(entropy_table.unsqueeze(0), self.precision).squeeze(0)
            self.entropy_table_cpu = self.entropy_table.cpu()

            for dist_i in range(self.num_distributions):
                assert torch.equal(self.entropy_table[dist_i], torchac._renorm_cast_cdf_(entropy_table[dist_i].unsqueeze(0).unsqueeze(0), self.precision).squeeze(0).squeeze(0))
            return True
        except NameError as e:
            logger.info(e)
            logger.info('update_entropy_table: renorm failed because missing torchac')
            return False

    def get_parameters(self, lr=None):
        assert lr is not None
        param_list = [
                {'params': self.Encoder.parameters(), 'name': 'encoder'},
                {'params': self.Decoder.parameters(), 'name': 'decoder'},
                {'params': self.bitEstimators.parameters(), 'lr': lr*10, 'name': 'bit_estimator'}]
        return param_list

    def entropy_encode(self, *args):
        bitstream, nbytes = self.entropy_encode_use_estim_prior(*args)
        logger.info('approx')
        logger.info(nbytes*8)
        return bitstream, nbytes
        _, nbytes, best_priors = self.entropy_encode_find_best_priors(*args)
        logger.info('find_best_priors:')
        logger.info(nbytes*8)

        return self.entropy_encode_use_estim_prior(args[0], best_priors)




    def encode(self, input_image, entropy_coding=True, out_fpath=None):
        '''
        encode an image, return bitstream as needed by decode
        egruns:
        python train.py --test_flags testKodakTiming --arch Balle2017ManyPriors --pretrain mse_4096_b2017manypriors_64pr_16px_swopt_cont
        '''
        nL = self.entropy_table.size(-1)
        feature = torch.round(self.Encoder(input_image))
        #feature = feature.clamp(self.min_feat, self.max_feat) # DBG FIX
        bs, nch, h, w = feature.shape
        ### LUT

        feat2enc = feature.view(nch,-1).transpose(0,1).unsqueeze(-1)#.contiguous().long()
        min_feat = feat2enc.min()
        max_feat = feat2enc.max()
        if min_feat < self.min_feat or max_feat > self.max_feat:
            print(f"encode: warning: min_feat={min_feat}, max_feat={max_feat} exceed the current CDF tables' range [{self.min_feat},{self.max_feat}]")
            print(f"Doubling the CDF table's size in this model's instance. This will increase runtime significantly; you likely want to clip the features instead.")
            self.min_feat = self.min_feat-self.max_feat
            self.max_feat *= 2
            print(f'New values: model.min_feat={self.min_feat}, model.max_feat={self.max_feat}')
            self.update_entropy_table()
            nL = self.entropy_table.size(-1)
        min_index = (-self.min_feat + min_feat).long()
        max_index = (-self.min_feat + max_feat + 1).long()
        feat_indices_lower = (feat2enc - min_feat).unsqueeze(0).expand(self.num_distributions, -1, -1, -1).long()
        feat_indices_upper = feat_indices_lower + 1

        entropy_table = self.entropy_table_float[...,min_index:max_index+1]

        cdf_tbl = entropy_table.unsqueeze(1).expand(self.num_distributions, h*w, nch, entropy_table.size(-1))
        probs = torch.gather(cdf_tbl, 3, feat_indices_upper) - torch.gather(cdf_tbl, 3, feat_indices_lower)

        _, indices = torch.sum(torch.clamp(-1.0* torch.log(probs + 1e-10) / math.log(2.0), 0, 50), dim=(2)).min(0)

        # replace min_feat by self.mean_feat when using full range of probabilities
        cdf_tbl = torch.gather(self.entropy_table.unsqueeze(0).expand(h*w,self.num_distributions, nch, nL), 1, indices.view(-1, 1, 1, 1).expand(h*w, 1, nch, nL)).squeeze(1).unsqueeze(0)

        features_int = ((feature.view(1, nch, -1).transpose(1,2)-self.min_feat) / self.q_intv).round().short().flatten()

        #breakpoint()
        #return [features_int, indices, h, w]
        if entropy_coding:
            bitstream = torchac.encode_cdf(cdf_tbl.cpu(), features_int.cpu())
        else:
            bitstream = features_int
        #return [feature, probs, features_int, None]
        # TODO combine into a light archive
        if out_fpath is not None and entropy_coding:
            with open(out_fpath+'.mbs', 'wb') as fp:
                fp.write(bitstream)
            #with open(out_fpath+'.h', 'w') as fp:
            #    fp.write(str(h))
            #with open(out_fpath+'.w', 'w') as fp:
            #    fp.write(str(w))
            utilities.compress_png(indices.view(1, h, w).numpy(), out_fpath+'.png')  # alternatively use compress_lzma

        return [bitstream, indices, h, w]
        # TODO accelerate // use LUT instead of bitEstimators


    def decode(self, bitstream, in_fpath=None):
        #return None  # FIXME RM


        if isinstance(bitstream, torch.Tensor):
            decoded_img = self.Decoder(bitstream)
            return decoded_img
        if in_fpath is not None:
            with open(in_fpath+'.mbs', 'rb') as fp:
                bitstream = fp.read()
            #with open(in_fpath+'.h', 'r') as fp:
            #    h = int(fp.read())
            #with open(in_fpath+'.w', 'r') as fp:
            #    w = int(fp.read())
            indices = torch.tensor(np.vstack(map(np.uint16, png.Reader(in_fpath+'.png').asDirect()[2])).astype(np.long))
            h, w = indices.shape
        else:
            bitstream, indices, h, w = bitstream
        nch = self.out_channel_M
        nL = self.entropy_table.size(-1)
        breakpoint()
        cdf_tbl = torch.gather(self.entropy_table_cpu.unsqueeze(0).expand(h*w,self.num_distributions, nch, nL), 1, indices.view(-1, 1, 1, 1).expand(h*w, 1, nch, nL)).squeeze(1).unsqueeze(0)
        features = torchac.decode_cdf(cdf_tbl.cpu(), bitstream).to(self.device)
        features = features.view(1, -1, nch).transpose(1,2).view(1,nch,h,w)
        features = (features * self.q_intv + self.min_feat).float()
        decoded_img = self.Decoder(features)
        return decoded_img#.cpu()

    def segment(self, input_image, output_fpath):
        '''
        egrun:
            python train.py --pretrain mse_4096_b2017manypriors_64pr_16px_adam_2upd_d --test_flags segmentation --device -1
Command Line Args:   --pretrain mse_4096_b2017manypriors_64pr_16px_adam_2upd_d --test_flags segmentation --device -1

        '''
        # encode image, get indices
        with torch.no_grad():
            im_shape = input_image.shape
            if len(input_image.shape) <= 3:
                input_image = input_image.unsqueeze(0)
            compressed_feature_renorm = torch.round(self.Encoder(input_image))

            probs = self.bitEstimators(compressed_feature_renorm+0.5) - self.bitEstimators(compressed_feature_renorm-0.5)
            dist_select = torch.sum(torch.clamp(- torch.log2(probs + 1e-10), 0, 50), dim=(2)).argmin(0)
        del(compressed_feature_renorm)
        del(probs)

        color_palette = torch.tensor(distinct_colors.get_color_palette(self.num_distributions), dtype=torch.int16)
        r = color_palette[:,0].view(-1,1,1).expand(self.num_distributions, dist_select.size(1), dist_select.size(2))
        g = color_palette[:,1].view(-1,1,1).expand(self.num_distributions, dist_select.size(1), dist_select.size(2))
        b = color_palette[:,2].view(-1,1,1).expand(self.num_distributions, dist_select.size(1), dist_select.size(2))



        segmented = torch.zeros(3, dist_select.size(1), dist_select.size(2))
        segmented[0] = torch.gather(r, 0, dist_select.long())
        segmented[1] = torch.gather(g, 0, dist_select.long())
        segmented[2] = torch.gather(b, 0, dist_select.long())
        
        pt_helpers.tensor_to_imgfile(segmented, output_fpath)





    def timing_analysis(self, input_image, entropy_coding=False):
        '''
        python train.py --test_flags timing --arch Balle2017ManyPriors --pretrain mse_4096_b2017manypriors_64pr_16px_swopt_cont --device -1
        egrun
        python train.py --pretrain mse_4096_b2017manypriors_64pr_16px_adam_2upd_d --test_flags timing --device -1

        the side string lzma compression is timed separately !
        '''
        self.update_entropy_table()


        nL = self.entropy_table.size(-1)
        timedict = {'nnenc': [], 'getindices_lut': [], 'make_cdf_tbl': [], 'nndec': [], 'torchac_enc': [], 'torchac_dec': []}#, 'getindices_nn': []}
        #_ = self.forward(input_image)  # time side string enc/dec
        try:
            bitstream, indices, h, w = self.encode(input_image)
        except NameError:
            logger.info('timing_analysis: whole encode timing not completed due to missing torchac library')
        for rep in range(50):
            with torch.no_grad():
                # one whole enc
                #tic = time.perf_counter()
                #timedict['enc'].append(time.perf_counter()-tic)

                # nnenc
                tic = time.perf_counter()
                feature = torch.round(self.Encoder(input_image))
                timedict['nnenc'].append(time.perf_counter()-tic)

                # get best indices
                tic = time.perf_counter()
                bs, nch, h, w = feature.shape
                #breakpoint()
                feat2enc = feature.view(nch,-1).transpose(0,1).unsqueeze(-1)#.contiguous().long()
                #feat2enc = feature.view(-1, nch).unsqueeze(-1) # TODO DBG/rm, incorrect res
                min_feat = feat2enc.min()
                max_feat = feat2enc.max()
                min_index = (-self.min_feat + min_feat).long()
                max_index = (-self.min_feat + max_feat + 1).long()
                feat_indices_lower = (feat2enc - min_feat).unsqueeze(0).expand(self.num_distributions, -1, -1, -1).long()
                feat_indices_upper = feat_indices_lower + 1
                entropy_table = self.entropy_table_float[...,min_index:max_index+1]
                cdf_tbl = entropy_table.unsqueeze(1).expand(self.num_distributions, h*w, nch, entropy_table.size(-1))
                probs = torch.gather(cdf_tbl, 3, feat_indices_upper) - torch.gather(cdf_tbl, 3, feat_indices_lower)
                #indices = torch.sum(torch.clamp(-torch.log(probs + 1e-10) / math.log(2.0), 0, 50), dim=(2)).argmin(0)
                indices = torch.sum(-torch.log2(probs), dim=2).argmin(0)
                #breakpoint()
                timedict['getindices_lut'].append(time.perf_counter()-tic)

                # make cdf tbl
                tic = time.perf_counter()
                cdf_tbl = torch.gather(self.entropy_table.unsqueeze(0).expand(h*w,self.num_distributions, nch, nL), 1, indices.view(-1, 1, 1, 1).expand(h*w, 1, nch, nL)).squeeze(1).unsqueeze(0)
                timedict['make_cdf_tbl'].append(time.perf_counter()-tic)

                try:
                    # entropy_encode
                    tic = time.perf_counter()
                    features_int = ((feature.view(1, nch, -1).transpose(1,2)-self.min_feat) / self.q_intv).round().short().flatten()
                    bitstream = torchac.encode_cdf(cdf_tbl, features_int)
                    timedict['torchac_enc'].append(time.perf_counter()-tic)

                    # entropy_decode
                    tic = time.perf_counter()
                    entropy_decoded = torchac.decode_cdf(cdf_tbl, bitstream)
                    timedict['torchac_dec'].append(time.perf_counter()-tic)

                    # nndec
                    tic = time.perf_counter()
                    reconst = self.Decoder(feature)
                    timedict['nndec'].append(time.perf_counter()-tic)
                except NameError:
                    # missing torchac
                    pass


                logger.info(timedict)


        for ares in timedict:
            timedict[ares] = statistics.mean(timedict[ares])
        logger.info('\n')
        logger.info(timedict)
        return timedict

    def forward(self, input_image):
        num_dists_to_force_train = 0

        im_shape = input_image.shape
        feature = self.Encoder(input_image)

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
        # recon_image = prediction + recon_res
        clipped_recon_image = recon_image.clamp(0., 1.)
        # distortion
        #visual_loss = torch.mean((recon_image - input_image).pow(2))
        visual_loss = self.lossfun(recon_image, input_image)


        total_bits_feature = 0

        if self.dist_patch_size == 1 and self.nchans_per_prior == self.out_channel_M:
            probs = self.bitEstimators(compressed_feature_entropy+0.5) - self.bitEstimators(compressed_feature_entropy-0.5)
            total_bits = torch.sum(torch.clamp(- torch.log2(probs + 1e-10), 0, 50), dim=(2))
            minbits, dist_select = total_bits.min(0)
            feature_batched_shape = compressed_feature_entropy.shape
            max_num_dists_to_force_train = int(compressed_feature_entropy.size(0)*compressed_feature_entropy.size(2)*compressed_feature_entropy.size(3)*0.1)

            #im_batched = compressed_feature_renorm
        else:
            im_batched = pt_ops.img_to_batch(compressed_feature_entropy, self.dist_patch_size, self.nchans_per_prior)
            feature_batched_shape = im_batched.shape
            probs = self.bitEstimators(im_batched+0.5) - self.bitEstimators(im_batched-0.5)
            total_bits = torch.sum(torch.clamp(- torch.log2(probs + 1e-10), 0, 50), dim=(2,3,4)) # per dist_i, batch
            minbits, dist_select = total_bits.min(0)
            max_num_dists_to_force_train = int(feature_batched_shape[0] * 0.1)

        #print(dist_select)
        used_dists = dist_select.unique()
        used_dists_cpu = used_dists.cpu()
        self.dists_last_use[used_dists.cpu()] = 0
        unused_dists = np.setdiff1d(np.arange(self.num_distributions), used_dists_cpu, assume_unique=True)
        self.dists_last_use[unused_dists] += 1
        if self.training and self.num_distributions > 1:# and len(used_dists) < min((self.num_distributions // 4*3), self.out_channel_M//4*3):
            dists_i_to_train = np.argwhere(self.dists_last_use>50).flatten()
            num_dists_to_force_train = min(dists_i_to_train.size, max_num_dists_to_force_train)

            victims = minbits.flatten().sort(descending=True).indices[:num_dists_to_force_train]
            dist_select.flatten()[victims] = torch.tensor(np.random.choice(unused_dists, victims.size(), replace=False), device=dist_select.device)

        if self.training or not self.entropy_coding:
            total_bits_feature = torch.gather(total_bits,0,dist_select.unsqueeze(0)).sum()
        else:
            total_bits_feature_theo = torch.gather(total_bits,0,dist_select.unsqueeze(0)).sum()
            logger.info('theobits')
            logger.info(total_bits_feature_theo)
            bitstream, nbytes = self.entropy_encode(compressed_feature_entropy, dist_select)
            total_bits_feature = nbytes * 8
            logger.info('actualbits')
            logger.info(total_bits_feature)
            #breakpoint()
        if self.training:
            bpp_sidestring = torch.tensor((feature_batched_shape[0]*math.log2(self.num_distributions)) / (im_shape[0] * im_shape[2] * im_shape[3]))
        else:
            bpp_sidestring = torch.tensor(pt_helpers.get_num_bits(dist_select.cpu(), integers=True), dtype=torch.float32) / (im_shape[0] * im_shape[2] * im_shape[3])
        bpp_feature = total_bits_feature / (im_shape[0] * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_sidestring
        return clipped_recon_image, visual_loss, bpp_feature, bpp_sidestring, bpp, used_dists_cpu.tolist(), num_dists_to_force_train


    def entropy_encode_find_best_priors(self, features, *args):
        '''
        encode one spatial location at a time checking for the best prior
        completely unnecessary
        '''
        bitstream = []
        dists = []
        nbytes = 0
        nbyteslist = []
        bs, nch, h, w = features.shape
        nL = self.entropy_table.size(-1)

        for bn in range(bs):
            features_int = features[bn].reshape(1,nch,-1).transpose(1,2)  # bs, h*w, nch
            features_int = ((features_int - self.min_feat) / self.q_intv).round().short()
            for adim in range(h*w):
                afeature = features_int[bn, adim]
                best_dist = 0
                cdf_tbl = self.entropy_table[0].unsqueeze(0).unsqueeze(0)
                smallest_bitstream = torchac.encode_cdf(cdf_tbl, afeature)
                for adist in range(1, self.num_distributions):
                    cdf_tbl = self.entropy_table[adist].unsqueeze(0).unsqueeze(0)  # 1, 1, nch, L
                    encoded_str = torchac.encode_cdf(cdf_tbl, afeature)
                    if len(smallest_bitstream) > len(encoded_str):
                        smallest_bitstream = encoded_str
                        best_dist = adist
                bitstream.append(smallest_bitstream)
                dists.append(best_dist)
                nbytes += len(smallest_bitstream)
                nbyteslist.append(len(smallest_bitstream))

        return bitstream, nbytes, torch.tensor(dists).reshape(h,w)

    def entropy_encode_use_estim_prior(self, features, dist_select):
        '''
        encodes the whole image at once using the theoretical best prior
        '''
        bitstream = []
        nbytes = 0
        bs, nch, h, w = features.shape
        nL = self.entropy_table.size(-1)
        for bn in range(bs):
            # too much memory to do all at once
            # entropy table: numdist*, nch, nL
            # dist_select: h, w
            # final dim: h*w, nch, nL
            cdf_tbl = torch.zeros(1, h*w, nch, nL, dtype=torch.short)
            dist_select_view = dist_select.reshape(bs, h*w)
            for adim in range(h*w):
                cdf_tbl[0, adim] = self.entropy_table[dist_select_view[bn, adim]]
            # // version
            # cdf_tbl = torch.gather(
            #     self.entropy_table.view(self.num_distributions, 1, nch, nL).expand(self.num_distributions, h*w, nch, nL).contiguous(),
            #     0,
            #     dist_select.view(1, h*w, 1, 1).expand(1, h*w, nch, nL)
            #     )
            features_int = features[bn].reshape(1,nch,-1).transpose(1,2).contiguous()  # bs, h*w, nch
            features_int = ((features_int - self.min_feat) / self.q_intv).round().short().flatten()

            encoded_str = torchac.encode_cdf(cdf_tbl, features_int)
            bitstream.append(encoded_str)
            nbytes += len(encoded_str)
            decoded_str = torchac.decode_cdf(cdf_tbl, encoded_str)
            #print(encoded_str)
            #print(z_int)
            #print(decoded_str)
            decoded_str = (decoded_str.view(1,h*w,nch).transpose(1,2).reshape(1,nch,h,w)*self.q_intv+self.min_feat).float()
            if features.max() > self.max_feat or features.min() < self.min_feat:
                logger.info('entropy_encode error: z out of range: [{}, {}] instead of [{}, {}]'.format(features.min(), features.max(), self.min_feat, self.max_feat))
                breakpoint()
            elif not torch.equal(features, decoded_str):
                logger.info('entropy_encode error: z and decoded_str do not match.')
                logger.info(cdf_tbl)
                breakpoint()

        logger.info(nbytes*8)
        return bitstream, nbytes

ManyPriors_ImageCompressor = Balle2017ManyPriors_ImageCompressor