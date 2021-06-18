# -*- coding: utf-8 -*-



'''
egruns:
test model with clic validation:100
python tests.py --encdec_dir --test_dpath ../../datasets/test/clic_valid_2020 --pretrain mse_4096_manypriors_64pr --arch ManyPriors --num_distributions 64 --lambda 4096
...
'''

import torch
import time
import numpy as np
import logging
import os
import statistics
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('..')
from common.libs import pt_helpers
from common.extlibs import pt_ms_ssim
from compression.libs import datasets
from compression.libs import initfun
from compression.libs import model_ops
from common.libs import utilities
from common.libs import pt_ops


logger = logging.getLogger("ImageCompression")

def parser_add_arguments(parser) -> None:
    parser.add_argument('--encode', type=str, help='Encode a given image file (dimensions have to be divisible by 16. Output is save_path/encoded/arg+.bitstream if not specified by out_fpath)')
    parser.add_argument('--decode', type=str, help='Decode a given bitstream. Output will be save_path/decoded/arg+.png if not specified')
    parser.add_argument('--segmentation', action='store_true', help='Segment images in the args.commons_test_dpath directory. Output will be save_path/segmentation/#.png')
    parser.add_argument('--plot', action='store_true', help='Plot cumulative distribution functions of a given (pretrain+params) model')
    parser.add_argument('--timing', type=str, help='Analyse timing of a given (pretrain) model using given image. (if arg is not an existing file then args.in_fpath is used)')
    parser.add_argument('--complexity', action='store_true', help='Analyze the complexity of a given model')
    parser.add_argument('--encdec_kodak', action='store_true', help='Test (encdec) args.test_dpath images')
    parser.add_argument('--encdec_dir', action='store_true', help='Test (encdec) args.test_dpath images, save results under test_<directory_name>')
    parser.add_argument('--encdec_commons', action='store_true', help='Test (encdec) args.test_commons_dpath images on CPU')
    parser.add_argument('--in_fpath', type=str, help='Input file path for timing when given arg is "default"')
    parser.add_argument('--out_fpath', help='Output file path for encode/decode (default: ../../models/compression/EXPNAME/encdec/IN_FPATH+EXT)')
    parser.add_argument('--test_commons_dpath', help='Path of Commons Test Photographs downloaded from https://commons.wikimedia.org/wiki/Category:Commons_Test_Photographs')
    parser.add_argument('--prior_utilisation', action='store_true', help='Measure utilisation of each CDF table with a given model and an image directory provided with --test_dpath')
    parser.add_argument('--get_kl_divergence', action='store_true', help='Measure Kullback–Leibler divergence between every distribution and its closest')

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
                if model.num_distributions == 8:
                    gs = fig.add_gridspec(2, 4, hspace=0,wspace=0)
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

def test_dir(model, step, jsonsaver, config: dict, loader, device, prefix='test', tb_logger=None, loss_cls=None, gt_src=['input', 'tuple'][0], incl_combined_loss=False):
    '''
    test a directory where images fit in GPU memory, s.a. kodak
    gt = 'input' if the loader returns one image which should be reconstructed as-is,
    gt = 'tuple' if the loader returns an input and a different ground-truth
    egrun:
        python tests.py --encdec_kodak --test_dpath "../../datasets/test/kodak/" --pretrain mse_4096_manypriors_64pr --arch ManyPriors --num_distributions 64
    '''
    with torch.no_grad():
        tmodel = model.eval().to(device)
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        sum_combined_loss = 0
        sum_visual_loss = 0
        sum_bpp_string = 0
        sum_bpp_side_string = 0
        sum_reconstruction_loss = 0
        sumTime = 0
        cnt = 0
        used_dists_all = set()
        if loss_cls is not None and hasattr(loss_cls, 'kernel'):
            orig_device = loss_cls.kernel.device
            loss_cls = loss_cls.to(device)
        for batch_idx, input in enumerate(loader):
            if gt_src == 'input':
                input = gt = input.to(device)
            elif gt_src == 'tuple':
                input, gt = input
                input = input.to(device)
                gt = gt.to(device)

            start_time = time.time()
            clipped_recon_image, visual_loss, bpp_feature, bpp_z, bpp, used_dists_now, _ = tmodel(input)
            if visual_loss is None:
                assert loss_cls is not None
                visual_loss = loss_cls(clipped_recon_image, gt)
            if device != torch.device('cpu'):
                pt_helpers.torch_cuda_synchronize()
            encoding_time = time.time()-start_time
            sumTime += encoding_time
            if used_dists_now is not None:
                used_dists_all.update(used_dists_now)
            visual_loss, bpp_feature, bpp_z, bpp = \
                torch.mean(visual_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)

            if config['test_cs']:
                if gt_src != 'input':
                    raise NotImplementedError
                test_batched_recon_img = pt_ops.img_to_batch(clipped_recon_image.detach(), config['test_cs'])
                test_batched_input_img = pt_ops.img_to_batch(input, config['test_cs'])
                mse = torch.nn.functional.mse_loss(test_batched_recon_img, test_batched_input_img, reduction='none').mean(dim=(1,2,3))
                psnr = 10 * (torch.log(1. / mse) / np.log(10))
                plt.hist(psnr.tolist(), 50, density=True, histtype='step', cumulative=True)
                plt.grid(True)

            mse = torch.nn.functional.mse_loss(clipped_recon_image.detach(), gt)
            psnr = 10 * (torch.log(1. / mse) / np.log(10))
            if gt_src != 'input':
                assert loss_cls is not None
                reconstruction_loss = loss_cls(input, clipped_recon_image)
                sum_reconstruction_loss += reconstruction_loss
            sumBpp += bpp
            sumPsnr += psnr
            try:
                msssim = pt_ms_ssim.ms_ssim(clipped_recon_image.detach(), gt, data_range=1.0, size_average=True)
                msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
            except AssertionError as e:
                print(e)
                msssim = 0
                msssimDB = 0
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            sum_combined_loss += config['train_lambda'] * visual_loss + bpp
            sum_visual_loss += visual_loss
            if gt == 'input':
               sum_reconstruction_loss = sum_visual_loss
            sum_bpp_string += bpp_feature
            sum_bpp_side_string += bpp_z
            logger.info("Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}, visual_loss: {:.6f}, enctime: {:.3f}, dists:{}".format(bpp, psnr, msssim, msssimDB, visual_loss, encoding_time, used_dists_now))
            test_savedir = os.path.join(config['save_path'], 'tests', str(step))
            os.makedirs(test_savedir, exist_ok=True)
            pt_helpers.tensor_to_imgfile(clipped_recon_image, os.path.join(
                test_savedir, prefix+str(batch_idx)+'.png'))
            cnt += 1
        plt.show()
        logger.info("Test on {} dataset: model-{}".format(prefix, step))
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        sum_combined_loss /= cnt
        sum_visual_loss /= cnt
        sum_bpp_string /= cnt
        sum_bpp_side_string /= cnt
        sum_reconstruction_loss /= cnt
        sumTime /= cnt
        logger.info("Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}, visual_loss:{:.6f}, combined_loss:{:.6f}, enctime: {:.6f}, dists: {}".format(sumBpp, sumPsnr, sumMsssim, sumMsssimDB, sum_visual_loss, sum_combined_loss, encoding_time, used_dists_all))
        if not config['nolog']:
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
                    '{}_combined_loss'.format(prefix): sum_combined_loss,
                    '{}_reconstruction_loss'.format(prefix): sum_reconstruction_loss},
                    write=False,
                    rm_none=True
                    )
            jsonsaver.add_res(
                step,
                {'{}_msssim'.format(prefix): sumMsssim,
                 '{}_msssimDB'.format(prefix): sumMsssimDB,
                 '{}_psnr'.format(prefix): sumPsnr},
                minimize=False
                )
        # ideally return a dict
        if loss_cls is not None and hasattr(loss_cls, 'kernel'):
            loss_cls = loss_cls.to(orig_device)
        if incl_combined_loss:
            return sum_visual_loss, sumBpp, sum_combined_loss
        return sum_visual_loss, sumBpp

def commons_test_photographs(model, step, jsonsaver, config: dict):
    '''Test on commons images. These images are typically too large to test on
    GPU (up to 4K) and the runtime is long so only manual evaluation is
    performed (with --test)
    TODO merge w/ test_dir
    egrun:
        python tests.py --encdec_commons --test_commons_dpath "../../datasets/test/Commons_Test_Photographs/" --pretrain ../../models/compression/mse_4096_manypriors_64pr/saved_models/checkpoint.pth --arch ManyPriors --num_distributions 64
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
            if config['test_cs']:
                input = pt_ops.img_to_batch(input, config['test_cs'])
            outdec_fpath = os.path.join(test_savedir, str(batch_idx)+'.png')

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

            if config['test_cs']:
                breakpoint()
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
        if not config['nolog']:
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
        python tests.py --decode ../../models/compression/checkpoints/mse_4096_manypriors_64pr/encoded/Garden_snail_moving_down_the_Vennbahn_in_disputed_territory_(DSCF5879).png" --pretrain mse_4096_manypriors_64pr --arch ManyPriors --num_distributions 64 --device -1
    '''
    model.update_entropy_table()
    decoded_tensor = model.decode(bitstream=None, in_fpath=in_fpath)
    pt_helpers.tensor_to_imgfile(decoded_tensor, out_fpath)

def prior_utilisation(model, test_dpath, save_path):
    '''
    egrun:
        python tests.py --prior_utilisation --arch ManyPriors --num_distributions 64 --pretrain mse_4096_b2017manypriors_64pr_16px_adam_2upd_d
        

    '''
    cpunet = model.cpu().eval()
    cpunet.update_entropy_table()
    test_dname = utilities.get_leaf(test_dpath)
    if os.path.isdir(save_path):
        save_path = os.path.join(save_path, f'prior_utilization_{test_dname}.csv')
        print(f'prior_utilisation: save_path={save_path}')
    dists_utilisation = 0
    with torch.no_grad():
        test_dataset = datasets.TestDirDataset(data_dir=test_dpath, resize=None, verbose=True, crop_to_multiple=16)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=False, num_workers=0)
        for batch_idx, input in enumerate(tqdm(test_loader)):
            _, indices, _, _ = cpunet.encode(input, entropy_coding=False)
            try:
                dists_utilisation += indices.squeeze().bincount(minlength=cpunet.num_distributions)
            except RuntimeError as e:
                print(f'prior_utilization encountered an error: {e}')
                breakpoint()
            with open(save_path, 'w') as fp:
                fp.write(','.join(str(int(i)) for i in dists_utilisation))
    print(f'prior_utilisation: final results writen to save_path={save_path}')
    # For some reason this ends up segfaulting, at ../../datasets/test/clic_test_pro/8ea6b5d7e5ec536504ec6c60a7c08c57.png?

def get_kl_distance(model, save_path):
    '''
    for each distribution, find another distribution with the smallest KL divergence
    eg:
    python tests.py --get_kl_divergence --arch ManyPriors --num_distributions 64 --pretrain mse_4096_b2017manypriors_64pr_16px_adam_2upd_d
    kl divergence: total=1294.6434326171875; list=[11.180265426635742, 15.62148666381836, 0.6328821182250977, 25.425758361816406, 22.071008682250977, 8.394054412841797, 14.890915870666504, 5.230834484100342, 32.154212951660156, 13.533382415771484, 16.60919189453125, 14.945428848266602, 21.37224006652832, 21.87444305419922, 23.744617462158203, 24.135114669799805, 33.01675033569336, 19.608814239501953, 44.85163116455078, 15.68785285949707, 8.100882530212402, 12.84747314453125, 15.888026237487793, 40.21651077270508, 40.77545928955078, 30.817012786865234, 16.717918395996094, 20.28795051574707, 34.27934265136719, 21.44320297241211, 21.49303436279297, 29.11978530883789, 8.077003479003906, 25.14607048034668, 39.531219482421875, 22.34889793395996, 0.3080233037471771, 8.447464942932129, 22.98062515258789, 25.84206771850586, 14.27016544342041, 20.396366119384766, 44.062286376953125, 19.36559295654297, 8.124574661254883, 28.762420654296875, 30.26300048828125, 7.108716011047363, 7.340500831604004, 32.1710319519043, 8.623671531677246, 8.779535293579102, 9.464139938354492, 22.005931854248047, 22.983409881591797, 25.63197135925293, 8.526240348815918, 21.092470169067383, 21.44297218322754, 28.9342098236084, 10.710474014282227, 32.67121124267578, 5.2777180671691895, 26.985843658447266], mean=20.228803634643555
    
    2 priors:
    python tests.py --get_kl_divergence --arch ManyPriors --num_distributions 2 --pretrain mse_4096_b2017manypriors_2pr_16px_adam_2upd
    
    kl divergence: total=923.2313232421875; list=[775.3006591796875, 147.93069458007812]; mean=461.61566162109375
    '''
    model.update_entropy_table()
    # CDF to PDF
    probs = model.entropy_table_float[:,:,1:] - model.entropy_table_float[:,:,:-1]
    probs = torch.max(probs, torch.zeros_like(probs)+0.00000001)  # avoid division by zero
    total_kl = 0
    kl_list = []
    for distn_a in range(model.num_distributions):
        dist_a = probs[distn_a]
        min_kl = None
        for distn_b in range(model.num_distributions):
            if distn_a == distn_b:
                continue
            dist_b = probs[distn_b]
            kl = (dist_a * torch.log2(dist_a / dist_b)).sum()
            if min_kl is None or min_kl > kl:
                min_kl = kl
        total_kl += min_kl
        kl_list.append(float(min_kl))
    # For each distribution, estimate cost to encode a symbol with the closest (next cheapest) distribution
    # estimate probabilities
    # for each distribution; get cheapest msg to encode
    # for each other distrubiton: calculate cost to encode that message and find min
    print(f'kl divergence: total={total_kl}; list={kl_list}; mean={total_kl/model.num_distributions}')
    # Baseline: 

if __name__ == "__main__":
    args, jsonsaver = initfun.get_args_jsonsaver(parser_add_arguments, parser_autocomplete)
    device = pt_helpers.get_device(args.device)

    model = model_ops.init_model(**vars(args))
    if args.pretrain is not None and not os.path.isfile(args.pretrain):
        args.pretrain = initfun.get_best_checkpoint(exp=args.pretrain, prefix=args.pretrain_prefix, step=args.global_step)
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
        #commons_loader = torch.utils.data.DataLoader(
        #    dataset=datasets.TestDirDataset(args.test_commons_dpath), shuffle=False, batch_size=1, num_workers=1)
        commons_test_photographs(model, step=global_step, jsonsaver=jsonsaver, config=vars(args))
    elif args.encdec_dir:
        test_set_name = utilities.get_leaf(args.test_dpath)
        test_loader = torch.utils.data.DataLoader(
            dataset=datasets.TestDirDataset(args.test_dpath, crop_to_multiple=64), shuffle=False, batch_size=1, num_workers=1)
        test_dir(model, step=global_step, jsonsaver=jsonsaver, config=vars(args), device=device, loader=test_loader, prefix='test_{}'.format(test_set_name))
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
    elif args.prior_utilisation:
        prior_utilisation(model, test_dpath=args.test_dpath, save_path=args.save_path)
    elif args.get_kl_divergence:
        get_kl_distance(model, save_path=args.save_path)
    else:
        logger.info('Nothing to do.')
        exit(-1)
