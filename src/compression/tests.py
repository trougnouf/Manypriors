# -*- coding: utf-8 -*-

import torch
import time
import numpy as np
import logging
import os
import statistics
import math
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from common.libs import pt_helpers
from common.extlibs import pt_ms_ssim
from compression.libs import datasets
from compression.libs import initfun
from compression.libs import model_ops
from common.libs import utilities


logger = logging.getLogger("ImageCompression")

def parser_add_arguments(parser) -> None:
    parser.add_argument('--encode', type=str, help='Encode a given image file (dimensions have to be divisible by 16. Output is save_path/encoded/arg+.bitstream if not specified by out_fpath)')
    parser.add_argument('--decode', type=str, help='Decode a given bitstream. Output will be save_path/decoded/arg+.png if not specified')
    parser.add_argument('--segmentation', action='store_true', help='Segment images in the args.commons_test_dpath directory. Output will be save_path/segmentation/#.png')
    parser.add_argument('--plot', action='store_true', help='Plot cumulative distribution functions of a given (pretrain+params) model')
    parser.add_argument('--timing', type=str, help='Analyse timing of a given (pretrain) model using given image. (if arg is not an existing file then args.in_fpath is used)')
    parser.add_argument('--complexity', action='store_true', help='Analyze the complexity of a given model')
    parser.add_argument('--encdec_kodak', action='store_true', help='Test (encdec) args.test_dpath images')
    parser.add_argument('--encdec_commons', action='store_true', help='Test (encdec) args.test_commons_dpath images on CPU')
    parser.add_argument('--in_fpath', type=str, help='Input file path for timing when given arg is "default"')
    parser.add_argument('--out_fpath', help='Output file path for encode/decode (default: checkpoints/EXPNAME/encdec/IN_FPATH+EXT)')
    parser.add_argument('--test_commons_dpath', help='Path of Commons Test Photographs downloaded from https://commons.wikimedia.org/wiki/Category:Commons_Test_Photographs')


def parser_autocomplete(args):
    args.pretrain_prefix = 'val' if args.val_dpath is not None else 'test'
    check_parameters(args)

def check_parameters(args):
    pass


def testComplexity(model, max_dim=4096):
    '''
    egrun:
        python tests.py --complexity --pretrain mse_4096_manypriors_64pr --arch ManyPriors --num_distributions 64
    '''
    model = model.cpu()
    model.update_entropy_table()
    with torch.no_grad():
        res = model.eval().complexity_analysis()
        logger.info(res)
        return res

def testTiming(model, in_fpath):
    '''
    egrun:
        python tests.py --timing "../../datasets/test/Commons_Test_Photographs" --pretrain mse_4096_manypriors_64pr --arch ManyPriors --num_distributions 64
    '''
    #test_img = datasets.TestDirDataset(data_dir=args.test_commons_dpath, resize=None, verbose=True)[6]
    model = model.cpu()
    test_img = pt_helpers.fpath_to_tensor(in_fpath).unsqueeze(0).cpu()
    with torch.no_grad():
        model.eval()
        timing_dict = model.timing_analysis(test_img)
        logger.info(timing_dict)
        return timing_dict

def testSegmentation(model, imgs_dpath, save_path):
    '''
    Segment the images in commons_test_dpath by distribution index.
    egrun:
        python tests.py --segmentation --commons_test_dpath "../../datasets/test/Commons_Test_Photographs" --pretrain mse_4096_manypriors_64pr --arch ManyPriors --num_distributions 64

    '''
    model.update_entropy_table()
    model = model.eval().cpu()
    #for testimgid in range(25):
    for testimgid, test_img in enumerate(datasets.TestDirDataset(data_dir=imgs_dpath, resize=None, verbose=True)):
        #test_img = datasets.TestDirDataset(data_dir=imgs_dpath, resize=None, verbose=True)#[testimgid]
        test_savedir = os.path.join(save_path, 'segment')
        os.makedirs(test_savedir, exist_ok=True)
        test_savepath = os.path.join(test_savedir, '{}.png'.format(testimgid))
        model.segment(test_img, test_savepath)

def visualizePriors(model, save_path=None):
    '''
    egrun:
        python tests.py --plot --pretrain mse_4096_manypriors_64pr --arch ManyPriors --num_distributions 64
    '''
    ONEROW = False
    TICK = True
    with torch.no_grad():
        model.eval()
        model.update_entropy_table()
        entropy_table = model.entropy_table_float.cpu()
        x = np.arange(model.min_feat-.5, model.max_feat+1.5)
        # plt.plot(x, entropy_table[0].transpose(0,1))

        fig = plt.figure()
        if model.num_distributions > 1:
            if ONEROW:
                gs = fig.add_gridspec(1, model.num_distributions, hspace=0,wspace=0)  # all flat
                axs = gs.subplots(sharex=True, sharey=True)
                dist = 0
                for axa in axs:
                    if TICK == False:
                        plt.xticks([]) # hide ticks
                        plt.yticks([]) # hide ticks
                    axa.plot(x, entropy_table[dist].transpose(0,1))
                    #axa.set_xlabel("CDF_{}".format(dist))
                    dist += 1
            else:
                gs = fig.add_gridspec(math.ceil(math.sqrt(model.num_distributions)), math.ceil(math.sqrt(model.num_distributions)), hspace=0,wspace=0)

                axs = gs.subplots(sharex=True, sharey=True)
                dist = 0
                try:
                    for axa in axs:
                        for axb in axa:
                            if TICK == False:
                                plt.xticks([]) # hide ticks
                                plt.yticks([]) # hide ticks
                            axb.plot(x, entropy_table[dist].transpose(0,1))
                            dist += 1
                    for axa in axs:
                        for axb in axa:
                            axb.label_outer()
                except TypeError as e:
                    print(e)
                    axs.plot(x, entropy_table.transpose(0,1))
                    axs.label_outer()
        else:
            # actually everything should work in the 1st condition for any ndists
            plt.figure()

            plt.plot(x, entropy_table.transpose(0,1))



        plt.show()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'priors.svg'))

def test_dir(model, step, jsonsaver, config: dict, loader, device, prefix='test', tb_logger=None):
    '''
    test a directory where images fit in GPU memory, s.a. kodak
    egrun:
        python tests.py --encdec_kodak --test_dpath "../../datasets/test/kodak/" --pretrain mse_4096_manypriors_64pr --arch ManyPriors --num_distributions 64
    '''
    with torch.no_grad():
        model.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        sum_combined_loss = 0
        sum_visual_loss = 0
        sum_bpp_string = 0
        sum_bpp_side_string = 0
        sumTime = 0
        cnt = 0
        used_dists_all = set()
        for batch_idx, input in enumerate(loader):
            input = input.to(device)
            # if config['consistent_patch_size']:
            #     input_bak = input
            #     input = pt_ops.img_to_batch(input, config['image_size'])
            #     output = torch.empty_like(input)
            #     cur_batch_i = 0
            #     start_time = time.time()
            #     visual_loss = []
            #     bpp_feature = []
            #     bpp_z = []
            #     bpp = []
            #     #used_dists = []
            #     while cur_batch_i < input.size(0):
            #         cur_batch_i_next = min(cur_batch_i+config['batch_size'], input.size(0))
            #         cur_batch = input[cur_batch_i: cur_batch_i_next]
            #         cur_output, cur_visual_loss, cur_bpp_feature, cur_bpp_z, cur_bpp, cur_used_dists_now, _ = model(cur_batch)

            #         output[cur_batch_i: cur_batch_i_next] = cur_output
            #         visual_loss.append(cur_visual_loss)
            #         bpp_feature.append(cur_bpp_feature)
            #         bpp_z.append(cur_bpp_z)
            #         bpp.append(cur_bpp)
            #         cur_batch_i = cur_batch_i_next
            #     clipped_recon_image = output.clamp(0., 1.)
            #     visual_loss = torch.tensor(visual_loss).mean()
            #     bpp_feature = torch.tensor(bpp_feature).mean()
            #     bpp_z = torch.tensor(bpp_z).mean()
            #     bpp = torch.tensor(bpp).mean()
            #     used_dists_now = None  # NotImplemented, add update to the loop above if required
            #     encoding_time = time.time()-start_time
            # else:
            start_time = time.time()
            clipped_recon_image, visual_loss, bpp_feature, bpp_z, bpp, used_dists_now, _ = model(input)
            if device != torch.device('cpu'):
                pt_helpers.torch_cuda_synchronize()
            encoding_time = time.time()-start_time
            sumTime += encoding_time
            if used_dists_now is not None:
                used_dists_all.update(used_dists_now)
            visual_loss, bpp_feature, bpp_z, bpp = \
                torch.mean(visual_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
            # if config['consistent_patch_size']:
            #     _, ch, height, width = input_bak.shape
            #     input = input_bak
            #     clipped_recon_image = pt_ops.batch_to_img(clipped_recon_image, height, width, ch=ch)
            mse = torch.nn.functional.mse_loss(clipped_recon_image.detach(), input)
            psnr = 10 * (torch.log(1. / mse) / np.log(10))
            sumBpp += bpp
            sumPsnr += psnr
            msssim = pt_ms_ssim.ms_ssim(clipped_recon_image.detach(), input, data_range=1.0, size_average=True)
            msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            sum_combined_loss += config['train_lambda'] * visual_loss + bpp
            sum_visual_loss += visual_loss
            sum_bpp_string += bpp_feature
            sum_bpp_side_string += bpp_z
            logger.info("Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}, visual_loss: {:.6f}, enctime: {:.3f}, dists:{}".format(bpp, psnr, msssim, msssimDB, visual_loss, encoding_time, used_dists_now))
            test_savedir = os.path.join(config['save_path'], 'tests', str(step))
            os.makedirs(test_savedir, exist_ok=True)
            pt_helpers.tensor_to_imgfile(clipped_recon_image, os.path.join(
                test_savedir, str(batch_idx)+'.png'))
            cnt += 1

        logger.info("Test on {} dataset: model-{}".format(prefix, step))
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        sum_combined_loss /= cnt
        sum_visual_loss /= cnt
        sum_bpp_string /= cnt
        sum_bpp_side_string /= cnt
        sumTime /= cnt
        logger.info("Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}, visual_loss:{:.6f}, combined_loss:{:.6f}, enctime: {:.6f}, dists: {}".format(sumBpp, sumPsnr, sumMsssim, sumMsssimDB, sum_visual_loss, sum_combined_loss, encoding_time, used_dists_all))
        if tb_logger is not None:
            logger.info("Add tensorboard---Step:{}".format(step))
            tb_logger.add_scalar("BPP_Test", sumBpp, step)
            tb_logger.add_scalar("PSNR_Test", sumPsnr, step)
            tb_logger.add_scalar("MS-SSIM_Test", sumMsssim, step)
            tb_logger.add_scalar("MS-SSIM_DB_Test", sumMsssimDB, step)
            tb_logger.add_scalar("RD_Test", sum_combined_loss, step)
            if len(used_dists_all) > 0:
                tb_logger.add_scalar("used_dists_test", len(used_dists_all), step)
        else:
            logger.info("No need to add tensorboard")
        jsonsaver.add_res(
                step,
                {'{}_bpp'.format(prefix): sumBpp,
                '{}_visual_loss.format(prefix)': sum_visual_loss,
                '{}_bpp_string'.format(prefix): sum_bpp_string,
                '{}_bpp_side_string'.format(prefix): sum_bpp_side_string,
                '{}_combined_loss'.format(prefix): sum_combined_loss},
                write=False
                )
        jsonsaver.add_res(
            step,
            {'{}_msssim'.format(prefix): sumMsssim,
             '{}_msssimDB'.format(prefix): sumMsssimDB,
             '{}_psnr'.format(prefix): sumPsnr},
            minimize=False
            )
        return sum_visual_loss, sumBpp

def commons_test_photographs(model, step, jsonsaver, config: dict):
    '''Test on commons images. These images are typically too large to test on
    GPU (up to 4K) and the runtime is long so only manual evaluation is
    performed (with --test)
    TODO merge w/ test_dir
    egrun:
        python tests.py --encdec_commons --test_commons_dpath "../../datasets/test/Commons_Test_Photographs/" --pretrain checkpoints/mse_4096_manypriors_64pr/saved_models/checkpoint.pth --arch ManyPriors --num_distributions 64
    '''
    cpunet = model.cpu().eval()
    with torch.no_grad():
        sumTime = 0
        cnt = 0
        bpps = []
        bpps_string = []
        bpps_side_string = []
        psnrs = []
        msssims = []
        combined_losses = []
        test_savedir = os.path.join(config['save_path'], 'commons_tests', str(step))
        used_dists_all = set()
        test_dataset = datasets.TestDirDataset(data_dir=config['test_commons_dpath'], resize=None, verbose=True)
        commons_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=False, num_workers=0)
        for batch_idx, input in enumerate(commons_loader):
            outdec_fpath = os.path.join(test_savedir, str(batch_idx)+'.png')
            #results[batch_idx] = dict()
            #input = input.to(device)
            # if config['consistent_patch_size']:
            #     raise NotImplementedError
            #     input_bak = input
            #     #input = pt_ops.img_to_batch(input, config['image_size'])
            #     output = torch.empty_like(input)
            #     cur_batch_i = 0
            #     start_time = time.time()
            #     visual_loss = []
            #     bpp_feature = []
            #     bpp_z = []
            #     bpp = []
            #     #used_dists = []
            #     while cur_batch_i < input.size(0):
            #         cur_batch_i_next = min(cur_batch_i+config['batch_size'], input.size(0))
            #         cur_batch = input[cur_batch_i: cur_batch_i_next]
            #         cur_output, cur_visual_loss, cur_bpp_feature, cur_bpp_z, cur_bpp, cur_used_dists_now, _ = net(cur_batch)

            #         output[cur_batch_i: cur_batch_i_next] = cur_output
            #         visual_loss.append(cur_visual_loss)
            #         bpp_feature.append(cur_bpp_feature)
            #         bpp_z.append(cur_bpp_z)
            #         bpp.append(cur_bpp)
            #         cur_batch_i = cur_batch_i_next
            #     clipped_recon_image = output.clamp(0., 1.)
            #     visual_loss = torch.tensor(visual_loss).mean()
            #     bpp_feature = torch.tensor(bpp_feature).mean()
            #     bpp_z = torch.tensor(bpp_z).mean()
            #     bpp = torch.tensor(bpp).mean()
            #     used_dists_now = None  # NotImplemented, add update to the loop above if required
            #     encoding_time = time.time()-start_time
            # else:
            start_time = time.time()

            clipped_recon_image, visual_loss, bpp_feature, bpp_z, bpp, used_dists_now, _ = cpunet(input)
            bpps_string.append(float(torch.mean(bpp_feature)))
            bpps.append(float(torch.mean(bpp)))
            bpps_side_string.append(float(bpp_z))


            # if device != torch.device('cpu'):
            #     pt_helpers.torch_cuda_synchronize()
            encoding_time = time.time()-start_time

            sumTime += encoding_time
            if used_dists_now is not None:
                used_dists_all.update(used_dists_now)
            # visual_loss, bpp_feature, bpp_z, bpp = \
            #     torch.mean(visual_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
            # if config['consistent_patch_size']:
            #     _, ch, height, width = input_bak.shape
            #     input = input_bak
            #     clipped_recon_image = pt_ops.batch_to_img(clipped_recon_image, height, width, ch=ch)
            mse = torch.nn.functional.mse_loss(clipped_recon_image.detach(), input)
            psnrs.append(float(10 * (torch.log(1. / mse) / np.log(10))))

            msssims.append(float(pt_ms_ssim.ms_ssim(clipped_recon_image.detach(), input, data_range=1.0, size_average=True)))

            combined_losses.append(float(config['train_lambda'] * visual_loss + bpp))

            os.makedirs(test_savedir, exist_ok=True)
            pt_helpers.tensor_to_imgfile(clipped_recon_image, outdec_fpath)
            cnt += 1
            print("PSNR: {}; MS-SSIM: {}, bpp: {}; Encoding time: {}".format(psnrs[-1], msssims[-1], bpps[-1], encoding_time))

        logger.info("Test on Kodak dataset: model-{}".format(step))

        sumTime /= cnt

        jsonsaver.add_res(
                step,
                {'commons_bpp': statistics.mean(bpps),
                'commons_bpp_string': statistics.mean(bpps_string),
                'commons_bpp_side_string': statistics.mean(bpps_side_string),
                'commons_combined_loss': statistics.mean(combined_losses)},
                write=False
                )
        jsonsaver.add_res(
            step,
            {'commons_msssim': statistics.mean(msssims),
             'commons_psnr': statistics.mean(psnrs)},
            minimize=False
            )
        jsonsaver.add_res(
            step,
            {'commons_bpps': bpps,
             'commons_psnrs': psnrs,
             'commons_msssims': msssims}, val_type=list)

def encode(model, in_fpath, out_fpath, device):
    '''
    egrun:
        python tests.py --encode "../../datasets/test/Commons_Test_Photographs/Garden_snail_moving_down_the_Vennbahn_in_disputed_territory_(DSCF5879).png" --pretrain mse_4096_manypriors_64pr --arch ManyPriors --num_distributions 64 --device -1
    '''
    model.update_entropy_table()  # this should be done before saving the model
    in_ptensor = pt_helpers.fpath_to_tensor(in_fpath).unsqueeze(0).to(device)
    model.encode(in_ptensor, entropy_coding=True, out_fpath=out_fpath)

def decode(model, in_fpath, out_fpath, device):
    '''
    egrun:
        python tests.py --decode "checkpoints/mse_4096_manypriors_64pr/encoded/Garden_snail_moving_down_the_Vennbahn_in_disputed_territory_(DSCF5879).png" --pretrain mse_4096_manypriors_64pr --arch ManyPriors --num_distributions 64 --device -1
    '''
    model.update_entropy_table()
    decoded_tensor = model.decode(bitstream=None, in_fpath=in_fpath)
    pt_helpers.tensor_to_imgfile(decoded_tensor, out_fpath)

if __name__ == "__main__":
    args, jsonsaver = initfun.get_args_jsonsaver(parser_add_arguments, parser_autocomplete)
    device = pt_helpers.get_device(args.device)

    model = model_ops.init_model(**vars(args))
    if args.pretrain is not None and not os.path.isfile(args.pretrain):
        args.pretrain = initfun.get_best_checkpoint(exp=args.pretrain, prefix=args.pretrain_prefix)
    if args.pretrain is not None:
        logger.info("loading model:{}".format(args.pretrain))
        global_step = model_ops.load_model(model, args.pretrain, device=device)

    if args.encode is not None:
        if not os.path.isfile(args.encode):
            args.encode = args.in_fpath
        if args.out_fpath is None:
            out_dpath = os.path.join(args.save_path, 'encoded')
            os.makedirs(out_dpath, exist_ok=True)
            args.out_fpath = os.path.join(out_dpath, utilities.get_leaf(args.encode))
        encode(model, args.encode, out_fpath=args.out_fpath, device=device)
    elif args.decode is not None:
        if args.out_fpath is None:
            out_dpath = os.path.join(args.save_path, 'decoded')
            os.makedirs(out_dpath, exist_ok=True)
            args.out_fpath = os.path.join(out_dpath, utilities.get_leaf(args.decode)+'.png')
        decode(model, args.decode, args.out_fpath, device=device)
    elif args.encdec_kodak:
        test_loader = torch.utils.data.DataLoader(
            dataset=datasets.TestDirDataset(args.test_dpath), shuffle=False, batch_size=1, num_workers=1)
        test_dir(model, step=global_step, jsonsaver=jsonsaver, config=vars(args), device=device, loader=test_loader, prefix='test')
    elif args.encdec_commons:
        commons_loader = torch.utils.data.DataLoader(
            dataset=datasets.TestDirDataset(args.test_commons_dpath), shuffle=False, batch_size=1, num_workers=1)
        commons_test_photographs(model, step=global_step, jsonsaver=jsonsaver, config=vars(args))
    elif args.complexity:
        testComplexity(model)
    elif args.timing is not None:
        if os.path.isfile(args.timing):
            args.in_fpath = args.timing
        testTiming(model, args.in_fpath)
    elif args.segmentation:
        testSegmentation(model, args.test_commons_dpath, args.save_path)
    elif args.plot:
        visualizePriors(model, save_path=args.save_path)
    else:
        logger.info('Nothing to do.')
        exit(-1)
