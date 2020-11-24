# -*- coding: utf-8 -*-
'''
Train a compression model. See configs/default.yaml for default parameters, libs/initfun.py:parser_add_argument for
parameters definition.
'''

import torch
import logging
import time
import os
import tensorboardX
import unittest
import sys
sys.path.append('..')
from common.extlibs import radam
from common.extlibs import pt_rangelars
from compression.libs import initfun
from compression.libs import Meter
from common.libs import locking
from compression.libs import model_ops
from compression import tests
from compression.tools import cleanup_checkpoints
from common.libs import pt_helpers
from compression.libs import datasets

OPTIMIZERS = {'RangeLars': pt_rangelars.RangerLars, 'RAdam': radam.RAdam, 'Adam': torch.optim.Adam}
logger = logging.getLogger("ImageCompression")

def parser_add_arguments(parser) -> None:
    # useful config
    parser.add_argument('--tot_step', type=int, help='Number of training steps')
    parser.add_argument('--reset_lr', action='store_true')
    parser.add_argument('--reset_optimizer', action='store_true')
    parser.add_argument('--reset_global_step', action='store_true')
    # moderately useful config
    parser.add_argument('--base_lr', type=float)
    parser.add_argument('--train_data_dpaths', nargs='*', type=str, help='Training image directories')
    # very unusual config
    parser.add_argument('--tot_epoch', type=int, help='Number of passes through the dataset')
    parser.add_argument('--lr_update_mode', help='use worse_than_previous')
    parser.add_argument('--lr_decay', type=float, help='LR is multiplied by this value whenever performance does not improve in an steps-epoch')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--print_freq', type=int, help='Logger frequency in # steps')
    parser.add_argument('--save_model_freq', type=int)
    parser.add_argument('--test_step', type=int)
    parser.add_argument('--optimizer_init', help='Initial optimizer: '+str(OPTIMIZERS.keys()))
    parser.add_argument('--optimizer_final', help='Final optimizer: '+str(OPTIMIZERS.keys()))
    parser.add_argument('--optimizer_switch_step', type=int, help='# of steps after which optimizer_final is chosen')

def parser_autocomplete(args):
    args.pretrain_prefix = 'val' if args.val_dpath is not None else 'test'
    check_parameters(args)

def check_parameters(args):
    assert args.optimizer_final is None or args.optimizer_final in OPTIMIZERS
    assert args.test_dpath is not None or args.val_dpath is not None, 'test_dpath and/or val_dpath required to update lr'

def train(model, train_loader, test_loader, val_loader, device, tb_logger, data_epoch, global_step, jsonsaver, optimizer, config):
    '''
    Train model for a data epoch
    returns current step
    '''
    logger.info("Data epoch {} begin".format(data_epoch))
    model.train()
    elapsed, losses, bpps, bpp_features, bpp_zs, visual_losses = [Meter.AverageMeter(config['print_freq']) for _ in range(6)]
    used_dists_all = set()

    previous_vis_loss = None
    previous_bpp_loss = None
    previous_loss = None
    # the following is pretty ugly, should use a queue or such
    preprevious_vis_loss = None
    preprevious_bpp_loss = None
    preprevious_loss = None
    test_prefix = 'test' if val_loader is None else 'val'
    val_test_loader = test_loader if val_loader is None else val_loader
    for batch_idx, input in enumerate(train_loader):
        input = input.to(device)
        locking.check_pause()
        start_time = time.time()
        global_step += 1
        # print("debug", torch.max(input), torch.min(input))
        clipped_recon_image, visual_loss, bpp_feature, bpp_z, bpp, used_dists_now, flag = model(input)

        distribution_loss = bpp
        if config['num_distributions'] <= 16 and used_dists_now is not None:
            used_dists_all.update(used_dists_now)

        distortion = visual_loss
        rd_loss = config['train_lambda'] * distortion + distribution_loss
        optimizer.zero_grad()
        rd_loss.backward()

        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)
        clip_gradient(optimizer, 5)
        optimizer.step()
        # model_time += (time.time()-start_time)


        if (global_step % config['print_freq']) == 0:
            # These were a separate step (global_step % cal_step), but we are
            # no longer calculating the average so this step should be simplified
            elapsed.update(time.time() - start_time)
            losses.update(rd_loss.item())
            bpps.update(bpp.item())
            bpp_features.update(bpp_feature.item())
            bpp_zs.update(bpp_z.item())
            visual_losses.update(visual_loss.item())

            # begin = time.time()
            tb_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)
            tb_logger.add_scalar('rd_loss', losses.avg, global_step)
            tb_logger.add_scalar('visual_loss', visual_losses.avg, global_step)
            #tb_logger.add_scalar('psnr', psnrs.avg, global_step)
            tb_logger.add_scalar('bpp', bpps.avg, global_step)
            tb_logger.add_scalar('bpp_feature', bpp_features.avg, global_step)
            tb_logger.add_scalar('bpp_z', bpp_zs.avg, global_step)
            process = global_step / config['tot_step'] * 100.0
            log = (' | '.join([
                f'{config["expname"]}',
                f'Step [{global_step}/{config["tot_step"]}={process:.2f}%]',
                f'Data Epoch {data_epoch}',
                f'Time {elapsed.val:.3f} ({elapsed.avg:.3f})',
                f'Lr {optimizer.param_groups[0]["lr"]}',
                f'Total Loss {losses.val:.3f} ({losses.avg:.3f})',
                #f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                f'Bpp {bpps.val:.5f} ({bpps.avg:.5f})',
                f'Bpp_feature {bpp_features.val:.5f} ({bpp_features.avg:.5f})',
                f'Bpp_z {bpp_zs.val:.5f} ({bpp_zs.avg:.5f})',
                f'Visual loss {visual_losses.val:.5f} ({visual_losses.avg:.5f})',
            ]))
            if used_dists_now is not None:
                if config['num_distributions'] <= 16:
                    log += ('| used_dists: {}'.format(used_dists_all))
                    used_dists_all = set()
                else:
                    log += ('| used_dists: {}'.format(len(used_dists_now)))
                    tb_logger.add_scalar("num_used_dists", len(used_dists_now), global_step)
                log += ('| flag: {}'.format(flag))

            logger.info(log)
        if (global_step % config['save_model_freq']) == 0:
            jsonsaver.add_res(
                global_step,
                {'train_bpp': bpps.avg,
                'train_visual_loss': visual_losses.avg,
                'train_bpp_string': bpp_features.avg,
                'train_bpp_side_string': bpp_zs.avg,
                'train_combined_loss': losses.avg,
                'lr_vis': optimizer.param_groups[0]['lr'],
                'lr_bpp': optimizer.param_groups[-1]['lr']}
                )
            model_ops.save_model(model, global_step, os.path.join(config['save_path'], 'saved_models'), optimizer=optimizer)

        if (global_step % config['test_step']) == 0:
            val_vis_loss, val_bpp_loss = tests.test_dir(model=model, step=global_step, jsonsaver=jsonsaver,
                                                       config=config, device=device, prefix=test_prefix,
                                                       loader=val_test_loader, tb_logger=tb_logger)
            model.train()
            update_vis = update_bit = False
            if previous_vis_loss is not None and previous_vis_loss < val_vis_loss and previous_loss < losses.avg and config['lr_update_mode'] != 'liujiaheng':
                if not config['two_worse_before_lr_update'] or (preprevious_vis_loss is not None and preprevious_vis_loss < val_vis_loss and preprevious_loss < losses.avg):
                    update_vis = True
            if previous_bpp_loss is not None and previous_bpp_loss < val_bpp_loss and previous_loss < losses.avg and config['lr_update_mode'] != 'liujiaheng':
                if not config['two_worse_before_lr_update'] or (preprevious_bpp_loss is not None and preprevious_bpp_loss < val_bpp_loss and preprevious_loss < losses.avg):
                    update_bit = True

            adjust_learning_rate(optimizer, global_step, lr_update_mode=config['lr_update_mode'],
                                 lr_decay=config['lr_decay'], bit=update_bit, encdec=update_vis)

            if config['two_worse_before_lr_update']:
                # TODO use a stack w/ any number of worse before update
                preprevious_vis_loss = previous_vis_loss
                preprevious_bpp_loss = previous_bpp_loss
                preprevious_loss = previous_loss
            previous_vis_loss = val_vis_loss
            previous_bpp_loss = val_bpp_loss
            previous_loss = losses.avg
        if (global_step % config['save_model_freq']) == 0:
            cleanup_checkpoints.cleanup_checkpoints(expname=config['expname'])

    jsonsaver.add_res(
        global_step,
        {'train_bpp': bpps.avg,
        'train_visual_loss': visual_losses.avg,
        'train_bpp_string': bpp_features.avg,
        'train_bpp_side_string': bpp_zs.avg,
        'train_combined_loss': losses.avg}
        )
    model_ops.save_model(model, global_step, os.path.join(config['save_path'], 'saved_models'), optimizer=optimizer)
    tests.test_dir(model=model, step=global_step, jsonsaver=jsonsaver, config=config, loader=val_test_loader, prefix=test_prefix, device=device, tb_logger=tb_logger)
    model.train()

    return global_step

# TODO check that pretrain is checked against None not ''

# TODO check args
def adjust_learning_rate(optimizer, global_step, lr_decay, lr_update_mode='worse_than_previous', bit=False, encdec=False):
    if lr_update_mode == 'worse_than_previous':
        for param_group in optimizer.param_groups:
            if ('bit' in param_group['name'] and not bit) or ('bit' not in param_group['name'] and not encdec):
                continue
            logger.info('adjust_learning_rate: {}: {}->{}'.format(
                param_group['name'],
                param_group['lr'], param_group['lr']*lr_decay))
            param_group['lr'] = param_group['lr'] * lr_decay
    else:
        raise NotImplementedError(lr_update_mode)

def reset_lr(optimizer, model, base_lr):
    model_parameters = model.get_parameters(lr=base_lr)
    for param_group in optimizer.param_groups:
        new_lr = None
        for model_parameter in model_parameters:
            if param_group['name'] == model_parameter['name']:
                if 'lr' in model_parameter:
                    new_lr = model_parameter['lr']
                else:
                    new_lr = base_lr
                continue
        if new_lr is None:
            new_lr = base_lr
        logger.info('reset_lr: reset lr of {} to {}'.format(param_group['name'], new_lr))
        param_group['lr'] = new_lr

def train_handler(args, jsonsaver, device):
    # get model, step
    global_step = 0
    model = model_ops.init_model(**vars(args))
    if args.pretrain is not None and not os.path.isfile(args.pretrain):
        args.pretrain = initfun.get_best_checkpoint(exp=args.pretrain, prefix=args.pretrain_prefix)
    if args.pretrain is not None:
        logger.info("loading model:{}".format(args.pretrain))
        if args.reset_global_step:
            model_ops.load_model(model, args.pretrain, device=device)
        else:
            global_step = model_ops.load_model(model, args.pretrain, device=device)

    # get test loader(s)
    val_loader, test_loader = datasets.get_val_test_loaders(args.val_dpath, args.test_dpath)

    # get optimizer
    optimizer = OPTIMIZERS[args.optimizer_init]
    is_init_optimizer = True
    if args.optimizer_final is not None and args.optimizer_switch_step <= global_step:
        optimizer = OPTIMIZERS[args.optimizer_final]
        print('optimizer: {}'.format(str(optimizer)))
        is_init_optimizer = False
    optimizer = optimizer(model.get_parameters(lr=args.base_lr), lr=args.base_lr)
    #optimizer = optim.Adam(parameters, lr=base_lr)
    if args.pretrain is not None and not args.reset_optimizer:
        logger.info("loading optimizer:{}".format(args.pretrain+'.opt'))
        model_ops.load_model(optimizer, args.pretrain+'.opt', device=device)
        # if os.path.isfile(args.pretrain+'.opt.module'):
        #     optimizer = torch.load(args.pretrain+'.opt.module', map_location=device)
        if args.reset_lr:
            reset_lr(optimizer, model, args.base_lr)
    if args.freeze_autoencoder or (args.freeze_autoencoder_steps is not None and args.freeze_autoencoder_steps > global_step):
        logger.info('Freezing autoencoder (experimental)')
        model.freeze_autoencoder()
    #global train_loader
    tb_logger = tensorboardX.SummaryWriter(os.path.join(args.save_path, 'events'))
    train_dataset = datasets.Datasets(args.train_data_dpaths, args.image_size)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=device.type != 'cpu',
                              num_workers=4)
    steps_epoch = global_step // (len(train_dataset) // (args.batch_size))
    for data_epoch in range(steps_epoch, args.tot_epoch):
        if args.optimizer_final is not None and args.optimizer_switch_step <= global_step:
            if is_init_optimizer:
                logger.info('Switching optimizer from {} to {}'.format(args.optimizer_init, args.optimizer_final))
                optimizer = OPTIMIZERS[args.optimizer_final](model.get_parameters(lr=args.base_lr), lr=args.base_lr)
                is_init_optimizer = False
        if global_step > args.tot_step:
            logger.info('Ending at global_step={}'.format(global_step))
            break
        if args.freeze_autoencoder_steps is not None and model.frozen_autoencoder and global_step >= args.freeze_autoencoder_steps:
            model.unfreeze_autoencoder()
            logger.info('unFreezing autoencoder')
            # def train(model, train_loader, test_loader, val_loader, device, tb_logger, data_epoch, global_step, jsonsaver, optimizer, config):

        global_step = train(model=model, data_epoch=data_epoch, global_step=global_step, jsonsaver=jsonsaver, optimizer=optimizer, config=vars(args), train_loader=train_loader, test_loader=test_loader, val_loader=val_loader, device=device, tb_logger=tb_logger)
        cleanup_checkpoints.cleanup_checkpoints(expname=args.expname)
        #save_model(model, global_step, save_path)


class Test_train(unittest.TestCase):
    '''
    [compression]$ python -m unittest discover .
    or
    [compression]$ python -m unittest train.py
    TODO use a subset of the dataset s.t. it doesn't take an unrealistic amount of time to run
    unittest yields "ResourceWarning: unclosed file <_io.BufferedReader" which doesn't occur in normal runs.
    '''
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_train_from_scratch_and_load(self):
        # train from scratch
        args, jsonsaver = initfun.get_args_jsonsaver(parser_add_arguments, parser_autocomplete,
                                                     ['--num_distributions', '64', '--arch', 'Balle2017ManyPriors', '--tot_step', '5000', '--train_lambda', '128', '--expname', 'unittest_scratch'])
        device = pt_helpers.get_device(args.device)
        train_handler(args, jsonsaver, device)
        self.assertTrue(os.path.isfile(os.path.join('checkpoints', 'unittest_scratch', 'trainres.json')))
        self.assertTrue(os.path.isfile(os.path.join('checkpoints', 'unittest_scratch', 'saved_models', 'iter_5000.pth')))
        # load that model
        # train from scratch
        args, jsonsaver = initfun.get_args_jsonsaver(parser_add_arguments, parser_autocomplete,
                                                     ['--num_distributions', '64', '--arch', 'Balle2017ManyPriors', '--tot_step', '5000', '--train_lambda', '64', '--pretrain', 'unittest_scratch', '--expname', 'unittest_load', '--reset_global_step'])
        device = pt_helpers.get_device(args.device)
        train_handler(args, jsonsaver, device)
        self.assertTrue(os.path.isfile(os.path.join('checkpoints', 'unittest_load', 'trainres.json')))
        self.assertTrue(os.path.isfile(os.path.join('checkpoints', 'unittest_load', 'saved_models', 'iter_5000.pth')))



if __name__ == "__main__":
    args, jsonsaver = initfun.get_args_jsonsaver(parser_add_arguments, parser_autocomplete)
    device = pt_helpers.get_device(args.device)
    train_handler(args, jsonsaver, device)
