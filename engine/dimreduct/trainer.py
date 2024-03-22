import os
from tqdm import tqdm
import torch
from torchvision.utils import save_image, make_grid
from engine.utils import AverageMeter
import pdb

def do_train(cfg, epoch, model, optimizer, dataloader, device, criterion, lr_scheduler=None):
    model.train()
    recon_losses = AverageMeter('recon_loss')
    
    for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        model.zero_grad()
        # forward pass
        recon_data = model(data)
        recon_loss = criterion(recon_data, target)
#         losses.update(loss.item(), data.size(0))  # data.size(0): Batch size
        recon_losses.update(recon_loss.item(), data.size(0))
        
        recon_loss.backward()
        optimizer.step()
        
        if batch_idx % cfg.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Recon Train Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f}) '
                  .format(epoch, batch_idx, len(dataloader), recon_loss=recon_losses,))


def do_val(cfg, epoch, model, dataloader, device, criterion):
    model.eval()
    recon_losses = AverageMeter('recon_loss')
    
    with torch.set_grad_enabled(False):
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)
            
            model.zero_grad()
            
            recon_data = model(data)
            recon_loss = criterion(recon_data, target)
            
            recon_losses.update(recon_loss.item(), data.size(0))
            
            if batch_idx % cfg.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Recon Val Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f}) '
                      .format(epoch, batch_idx, len(dataloader), recon_loss=recon_losses,))
            
    return recon_losses.avg


def inference(cfg, epoch, model, dataloader, device, criterion):
    model.eval()
    recon_losses = AverageMeter('recon_loss')
    
    with torch.set_grad_enabled(False):
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)
            
            model.zero_grad()
            
            recon_data = model(data)
            recon_loss = criterion(recon_data, target)
            
            recon_losses.update(recon_loss.item(), data.size(0))
            # os.makedirs(f'figures/{cfg.dataset}/batch{batch_idx}', exist_ok=True)
            # for i in range(len(data)):
            #     save_image(recon_data[i], f'figures/{cfg.dataset}/batch{batch_idx}/recon{i}.png')
            #     save_image(target[i], f'figures/{cfg.dataset}/batch{batch_idx}/target{i}.png')
            # break
            if batch_idx % cfg.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Recon Test Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f}) '
                      .format(epoch, batch_idx, len(dataloader), recon_loss=recon_losses,))
            
    return recon_losses.avg


def inference_patch(cfg, epoch, model, dataloader, device, criterion):
    model.eval()
    # recon_losses = AverageMeter('recon_loss')
    
    with torch.set_grad_enabled(False):
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)
            
            model.zero_grad()
            os.makedirs(f'figures/{cfg.dataset}/batch{batch_idx}', exist_ok=True)
            for pidx in range(10):
                recon_data = model(data, patchIdx=pidx)
                for i in range(len(data)):
                    save_image(recon_data[i], f'figures/{cfg.dataset}/batch{batch_idx}/recon{i}_patch{pidx}.png')
                    save_image(target[i], f'figures/{cfg.dataset}/batch{batch_idx}/target{i}.png')
            break
            # recon_loss = criterion(recon_data, target)
            
            # recon_losses.update(recon_loss.item(), data.size(0))
            
#             if batch_idx % cfg.print_freq == 0:
#                 print('Epoch: [{0}][{1}/{2}]\t'
#                       'Recon Test Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f}) '
#                       .format(epoch, batch_idx, len(dataloader), recon_loss=recon_losses,))
            
    # return recon_losses.avg
