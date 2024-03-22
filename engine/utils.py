import torch
import numpy as np
# import pdb

class PredAnalyst:
    def __init__(self, cfg):
        """Tracks and analyzes statistics of prediction"""
        self.sceneID_list = []
        self.frameID_list = []
        self.nodeID_list = []
        # self.class_list = []
        self.gt_state_list = []
        self.pred_state_list = []  # radial distance to the camera
        self.pred_metric_lists = {} # dict of list
        cfg = cfg
        for name in cfg.test.metrics:
            self.pred_metric_lists[name] = []


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def eval_metrics(names: list):
    metrics = {}
    for name in names:
        metrics[name] = AverageMeter(name)
    return metrics

        
def print_info(epoch, model, optimizer, loss_dict, logger):
    info = "Epoch:{},\t lr:{:6},\t".format(epoch, optimizer.param_groups[0]['lr']) 
    for k in loss_dict.keys():
        info += " {}:{:.4f},\t".format(k, loss_dict[k]) 
    if 'grad_norm' in loss_dict:
        info += ", \t grad_norm:{:.4f}".format(loss_dict['grad_norm'])
    
    if hasattr(logger, 'log_values'):
        logger.info(info)
        logger.log_values(loss_dict)#, step=max_iters * epoch + iters)
    else:
        print(info)


def to_world_coords(homog, loc):
    """Given H and (u, v) in image coordinates, returns world coordinates."""
    locHomogenous = np.hstack((loc, np.ones((loc.shape[0], 1))))
    loc_tr = np.transpose(locHomogenous)
    loc_tr = np.matmul(homog, loc_tr)
    locXYZ = np.transpose(loc_tr / loc_tr[2])
    return locXYZ[:, :2]

def post_process(cfg, X_global, y_global, pred_traj, frame_res=None, homog_files=None, pred_goal=None, dist_traj=None, dist_goal=None):
    '''post process the prediction output'''
    if len(pred_traj.shape) == 4:
        batch_size, T, K, dim = pred_traj.shape
    else:
        batch_size, T, dim = pred_traj.shape
    X_global = X_global.detach().to('cpu').numpy()
    y_global = y_global.detach().to('cpu').numpy()
    # if pred_goal is not None:
    #     pred_goal = pred_goal.detach().to('cpu').numpy()
    pred_traj = pred_traj.detach().to('cpu').numpy()
    if torch.is_tensor(frame_res):
        frame_res[0] = frame_res[0].detach().to('cpu').numpy()
        frame_res[1] = frame_res[1].detach().to('cpu').numpy()
    # if hasattr(dist_traj, 'mus'):
    #     dist_traj.to('cpu')
    #     dist_traj.squeeze(1)
    # if hasattr(dist_goal, 'mus'):
    #     dist_goal.to('cpu')
    #     dist_goal.squeeze(1)
    if dim == 4:
        pass# BBOX: denormalize and change the mode
    elif dim == 2:
        for i in range(len(pred_traj)):
            if len(pred_traj.shape) == 3:
                if cfg.dataset.normalize == 'zero-one':  #Normalize agent states
                    pred_traj[i, :, 0], pred_traj[i, :, 1] = pred_traj[i, :, 0]*frame_res[0][i], pred_traj[i, :, 1]*frame_res[1][i]
                else:
                    pred_traj[i, :, 0], pred_traj[i, :, 1] = pred_traj[i, :, 0]*frame_res[0][i]/cfg.model.convae.image_size, pred_traj[i, :, 1]*frame_res[1][i]/cfg.model.convae.image_size
                if homog_files[i] is not None:
                    pred_traj[i, :, :2] = to_world_coords(np.loadtxt(homog_files[i]), pred_traj[i, :, :2])
            elif len(pred_traj.shape) == 4:
                if cfg.dataset.normalize == 'zero-one':  #Normalize agent states
                    pred_traj[i, :, :, 0], pred_traj[i, :, :, 1] = pred_traj[i, :, :, 0]*frame_res[0][i], pred_traj[i, :, :, 1]*frame_res[1][i]
                    if cfg.dataset.name == 'sdd':
                        y_global[i, :, 0], y_global[i, :, 1] = y_global[i, :, 0]*frame_res[0][i], y_global[i, :, 1]*frame_res[1][i]
                else:
                    pred_traj[i, :, :, 0], pred_traj[i, :, :, 1] = pred_traj[i, :, :, 0]*frame_res[0][i]/cfg.model.convae.image_size, pred_traj[i, :, :, 1]*frame_res[1][i]/cfg.model.convae.image_size
                    if cfg.dataset.name == 'sdd':
                        y_global[i, :, 0], y_global[i, :, 1] = y_global[i, :, 0]*frame_res[0][i]/cfg.model.convae.image_size, y_global[i, :, 1]*frame_res[1][i]/cfg.model.convae.image_size
                assert pred_traj.shape[2]==K
                if homog_files[i] is not None:
                    for k in range(pred_traj.shape[2]):
                        pred_traj[i, :, k, :2] = to_world_coords(np.loadtxt(homog_files[i]), pred_traj[i, :, k, :2])

    return X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal
