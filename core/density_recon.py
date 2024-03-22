import os
import sys
sys.path.append(os.path.realpath('.'))
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from torch import optim
# from datasets import DensityDataset
from datasets import DensitySDD
from models import CAE
from engine.dimreduct.trainer import do_train, do_val, inference
import time
import pickle 


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--datalist_root", type=str, default='')
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--img_path", type=str, default='')
    parser.add_argument("--dataset", type=str, default='', help='holdout dataset')
    parser.add_argument("--batch_size", type=int,)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--max_epoch',type=int)
    parser.add_argument('--weight_decay',type=float,default= 1e-4)
    parser.add_argument('--lr',type=float,)
    parser.add_argument('--print_freq',type=int,default=10) 
    parser.add_argument('--ckpt_dir',type=str,default='checkpoints/cae') 
    parser.add_argument('--inference',default=False, action='store_true',)
    parser.add_argument('--out_dir',type=str,default='') 
    parser.add_argument('--test_epoch',type=int,)  
    parser.add_argument('--criterion', type=str, default='bce', choices=['bce', 'mse'])
    parser.add_argument('--kernel_size', type=str, )
    parser.add_argument('--const_sigma', type=int, )
    return parser.parse_args()    


def main():
    args = parse_args()
    for k,v in vars(args).items():
        print(k,": ",v)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Define an autoencoder model
    cae = CAE(in_channel=1)
    cae = cae.to(args.device)
    
    if args.criterion == 'bce':
        criterion = nn.BCELoss()
    elif args.criterion == 'mse':
        criterion = nn.MSELoss()
    
    if args.inference:  # test mode
        if args.dataset == 'sdd':
            test_dataset = DensitySDD(args, split='test', 
                           # transform=transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                           #           transforms.ToTensor()]),
                           )
        else:
            test_dataset = DensityDataset(args, split='test', 
                           transform=transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                     transforms.ToTensor()]),
                           )
        # load test dataloader
        test_dataloader = data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        print(f'length of testset: {len(test_dataloader.dataset)}')

        save_checkpoint_dir = os.path.join(args.ckpt_dir, args.dataset, args.out_dir) 
        print(save_checkpoint_dir)
        epoch = args.test_epoch
        cae.load_state_dict(torch.load(os.path.join(save_checkpoint_dir, 'Epoch_{}.pth'.format(str(epoch).zfill(3)))))
        
        # inference_patch(args, epoch, cae, test_dataloader, args.device, criterion)
        eval_results = inference(args, epoch, cae, test_dataloader, args.device, criterion)
        print(f'test loss for the (best) {epoch} epoch: {eval_results}')
        
    else:
        # load train density dataset
        if args.dataset == 'sdd':
            train_dataset = DensitySDD(args, split='train', 
                           # transform=transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                           #                               # transforms.RandomRotation(degrees=(0, 360)),
                           #                               transforms.ToTensor()]),
                           )
        else:
            train_dataset = DensityDataset(args, split='train', 
                           transform=transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                                         # transforms.RandomRotation(degrees=(0, 360)),
                                                         transforms.ToTensor()]),
                           )
        train_dataloader = data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        print(f'length of trainset: {len(train_dataloader.dataset)}')
        # load val density dataset 
        if args.dataset == 'sdd':
            val_dataset = DensitySDD(args, split='val', 
                           # transform=transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                           #                               # transforms.RandomRotation(degrees=(0, 360)),
                           #                               transforms.ToTensor()]),
                           )
        else:
            val_dataset = DensityDataset(args, split='val', 
                           transform=transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                                         # transforms.RandomRotation(degrees=(0, 360)),
                                                         transforms.ToTensor()]),
                           )
        val_dataloader = data.DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        print(f'length of valset: {len(val_dataloader.dataset)}')

        optimizer = optim.Adam(cae.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        val_loss_best = float('inf')
        epoch_model_best = 0
        save_checkpoint_dir = os.path.join(args.ckpt_dir, args.dataset, time.strftime("%d%b%Y-%Hh%Mm%Ss"))  # training outputs
        if not os.path.exists(save_checkpoint_dir):
            os.makedirs(save_checkpoint_dir)
        with open(os.path.join(save_checkpoint_dir, 'args.pkl'), 'wb') as f:
            pickle.dump(args, f)
        # Start training
        timestart = time.time()
        for epoch in range(args.max_epoch):

            print('\n*** Start Training *** Epoch: [%d/%d]\n' % (epoch + 1, args.max_epoch))
            do_train(args, epoch+1, cae, optimizer, train_dataloader, args.device, criterion, )

            val_loss = do_val(args, epoch+1, cae, val_dataloader, args.device, criterion)
            if val_loss_best>val_loss:
                val_loss_best=val_loss
                epoch_model_best=epoch
                torch.save(cae.state_dict(), os.path.join(save_checkpoint_dir, 'Epoch_{}.pth'.format(str(epoch).zfill(3))))

            with open(os.path.join(save_checkpoint_dir, "log.txt"),"a") as f:
                f.write(f'best val loss: {val_loss_best}, best epoch: {epoch_model_best}')

        print( f'best val loss: {val_loss_best}, best epoch: {epoch_model_best}' )     
        print('Total Training Time: %.4f' % (time.time() - timestart))

    
if __name__ == '__main__':
    main()
 