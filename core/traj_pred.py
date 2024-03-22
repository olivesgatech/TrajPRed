import os
import sys
sys.path.append(os.path.realpath('.'))
sys.path.append('../Trajectron-plus-plus/experiments/pedestrians/')
from get_stats_sdd import AverageMeter
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import random
import shutil
import time
import pickle
from datasets import make_dataloader
from models import make_model
from engine.trajpredict.trainer import do_train, do_val, inference
from utils.scheduler import ParamScheduler, sigmoid_anneal
from utils.logger import Logger
import logging
import argparse
import time
from configs import cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument("--config_file", default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument('--seed', help='manual seed to use, default is 2022', type=int, default=2022)
    args = parser.parse_args()
    
    return args

def build_optimizer(cfg, model):
    # optimizer = optim.RMSprop(all_params, lr=cfg.SOLVER.LR)
    if cfg.solver.optimizer == 'Adam':
    	optimizer = optim.Adam(model.parameters(), lr=cfg.solver.lr, weight_decay=cfg.solver.weight_decay)
    elif cfg.solver.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg.solver.lr, momentum=cfg.solver.momentum, weight_decay=cfg.solver.weight_decay)
        
    return optimizer

def main():
    args = parse_args()
    ## ====================== load configs ======================
    cfg.merge_from_file(args.config_file)  ## merge from unknown list of congigs?
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    ## ====================== setting random seed ======================
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f'using seed: {args.seed}')
        cfg.seed = args.seed
    cfg.model.frame_pred = cfg.dataset.frame_pred 
    ## ====================== build model, optimizer and scheduler ======================
    model = make_model(cfg)
    model = model.to(cfg.device)
    if cfg.model.convae.load_pretrain_weights:
        # load and freeeze the weights of convAE!
        cae_ckpt_dir = os.path.join(cfg.model.convae.ckpt_dir, cfg.dataset.name, cfg.model.convae.out_dir) 
        model.cae.load_state_dict(torch.load(os.path.join(cae_ckpt_dir, 'Epoch_{}.pth'.format(str(cfg.model.convae.test_epoch).zfill(3)))))
        print(f'loaded convae checkpoint from: {cae_ckpt_dir}/Epoch_{(str(cfg.model.convae.test_epoch).zfill(3))}.pth')
        if cfg.model.convae.pretrain:
            for name, param in model.cae.named_parameters():
                param.requires_grad = False

    print('Model built!')
    if cfg.method=='tprn_re_np': 
        # hyperparameter scheduler
        model.param_scheduler = ParamScheduler()
        model.param_scheduler.create_new_scheduler(
                                            name='kld_weight',
                                            annealer=sigmoid_anneal,
                                            annealer_kws={
                                                'device': cfg.device,
                                                'start': 0,
                                                'finish': 100.0,
                                                'center_step': 400.0,
                                                'steps_lo_to_hi': 100.0, 
                                            })
    optimizer = build_optimizer(cfg, model)
    print('optimizer built!')
    if cfg.solver.scheduler == 'exp':  # exponential schedule
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.solver.gamma)
    elif cfg.solver.scheduler == 'plateau':  # Plateau scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg.solver.lr_decay_rate, patience=cfg.solver.patience,
                                                            min_lr=5e-06, verbose=1)
    elif cfg.solver.scheduler == 'mslr':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.solver.lr_steps, gamma=cfg.solver.lr_decay_rate)                                 
    print('Schedulers built!')
    ## ====================== logger ======================
    if cfg.use_wandb:
        logger = Logger(cfg.method,
                        cfg,
                        project = cfg.project,
                        viz_backend="wandb"
                        )
    else:
        logger = logging.Logger(cfg.method)
    ## ====================== train, val, test engines ======================
    if cfg.test.inference:  # test mode
        # load test dataloader
        test_dataloader = make_dataloader(cfg, 'test')
        save_checkpoint_dir = os.path.join(cfg.ckpt_dir, cfg.dataset.name, f'seed{args.seed}', cfg.out_dir) 
        epoch = cfg.test.epoch
        model.load_state_dict(torch.load(os.path.join(save_checkpoint_dir, 'Epoch_{}.pth'.format(str(epoch).zfill(3)))))
        eval_results = inference(cfg, epoch, model, test_dataloader, cfg.device, logger=logger,)
        print(f'results for the (best) {epoch} epoch: ')
        results_file = os.path.join(save_checkpoint_dir, 'results_{}.pth'.format(str(epoch).zfill(3)))
        if not os.path.exists(results_file):
            pickle.dump(eval_results, open(results_file,'wb'))
    else:
        # load train / val dataloaders
        train_dataloader = make_dataloader(cfg, 'train')
        val_dataloader = make_dataloader(cfg, 'val')
        print('Dataloader built!')
        save_checkpoint_dir = os.path.join(cfg.ckpt_dir, cfg.dataset.name, f'seed{args.seed}', time.strftime("%d%b%Y-%Hh%Mm%Ss"))  # training outputs
        if not os.path.exists(save_checkpoint_dir):
            os.makedirs(save_checkpoint_dir)
        shutil.copy(args.config_file, os.path.join(save_checkpoint_dir, 'configs.yml'))    
        
        val_loss_best = float('inf')
        epoch_model_best = 0
    
        # Start training
        timestart = time.time()
        for epoch in range(cfg.solver.max_epoch):
            logger.info("Epoch:{}".format(epoch))
            do_train(cfg, epoch, model, optimizer, train_dataloader, cfg.device, logger=logger, lr_scheduler=lr_scheduler)

            val_loss = do_val(cfg, epoch, model, val_dataloader, cfg.device, logger=logger)
            if val_loss_best>val_loss:
                val_loss_best=val_loss
                epoch_model_best=epoch
                torch.save(model.state_dict(), os.path.join(save_checkpoint_dir, 'Epoch_{}.pth'.format(str(epoch).zfill(3))))

            # update LR
            if cfg.solver.scheduler == 'plateau':
                lr_scheduler.step(val_loss)
            elif cfg.solver.scheduler == 'mslr':
                lr_scheduler.step()
            # save the best model based on the lowest validation loss
            with open(os.path.join(save_checkpoint_dir, "log.txt"),"a") as f:
                f.write(f'best val loss: {val_loss_best}, best epoch: {epoch_model_best}')
        print( f'best val loss: {val_loss_best}, best epoch: {epoch_model_best}' )     
        print('Total Training Time: %.4f' % (time.time() - timestart))

if __name__ == '__main__':
    main()
        