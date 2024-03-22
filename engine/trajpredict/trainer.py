import os
import torch
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
from utils.visualization import Visualizer
from .evaluate import evaluate_multimodal, compute_kde_nll
from engine.utils import PredAnalyst, eval_metrics, post_process, print_info
from datasets.trajectory.preprocessing import restore
from models.trajpredict.model_utils import ModeKeys
import pandas as pd
from pyarrow import feather as pf
from torch.utils.data import DataLoader
# import pdb


def do_train(cfg, epoch, model, optimizer, dataloader, device, logger=None, lr_scheduler=None):
    model.train()
    # max_iters = len(dataloader)
    if cfg.dataset.name in ['eth', 'hotel', 'univ', 'zara1', 'zara2' ,'sdd']:
        viz = Visualizer(mode='plot')
    else:
        viz = Visualizer(mode='image')

        
    with torch.set_grad_enabled(True):
        for iters, batch in enumerate(tqdm(dataloader)):
            # For ETH_UCY and SDD datasets only
            if cfg.dataset.name in ['eth', 'hotel', 'univ', 'zara1', 'zara2', 'sdd']:
                input_x = batch['input_x_st']
                target_y = batch['target_y_st']
                input_imgs = batch['input_imgs']
                target_imgs = batch['target_imgs']
                if cfg.model.point_rel_enable:
                    neighbors_x = batch['neighbors_x_st']
                else:
                    neighbors_x = None
                # adjacency = restore(batch['neighbors_adjacency'])
                first_history_indices = batch['first_history_index']
                # if cfg.dataset.normalize == 'zero-one':
                input_x_frame = batch['input_x_frame']
                # else:
                #     input_x_frame = batch['input_x_st'][:,:,:cfg.model.motion.dec_output_dim]
            else:
                input_x = None
                # neighbors_st, adjacency, first_history_indices = None, None, None

            loss_dict, _ = model(input_x=input_x, input_imgs=input_imgs, input_x_frame=input_x_frame, target_y=target_y, target_imgs=target_imgs, neighbors_x=neighbors_x, first_history_indices=first_history_indices, mode = ModeKeys.TRAIN)
            if cfg.method == 'tprn_re_np': 
                if cfg.model.latent_dist == 'categorical':
                    loss = loss_dict['loss_goal'] + \
                           loss_dict['loss_traj'] + \
                           model.param_scheduler.kld_weight * loss_dict['loss_kld'] - \
                           1. * loss_dict['mutual_info_p']
                elif cfg.model.latent_dist == 'gaussian':
                    loss = loss_dict['loss_goal'] + \
                           loss_dict['loss_traj'] + \
                           model.param_scheduler.kld_weight * loss_dict['loss_kld']
                else:
                    loss = loss_dict['loss_traj']

                if cfg.model.discrepancy: loss += loss_dict['loss_disc'] * cfg.loss.weight_disc
                if cfg.model.convae.loss != 'null': loss += loss_dict['loss_cae']
                # if cfg.model.best_of_many_back: loss += loss_dict['loss_goal_back'] # back goal estimation
                model.param_scheduler.step()
            # for k, v in loss_dict.items():
            #     loss += v
            else:
                loss = loss_dict['loss_cae'] + loss_dict['loss_traj'] 
            loss_dict = {k:v.item() for k, v in loss_dict.items()}
            
            optimizer.zero_grad() 
            loss.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            
            if cfg.solver.scheduler == 'exp':
                lr_scheduler.step()
            if iters % cfg.print_interval == 0:
                print_info(epoch, model, optimizer, loss_dict, logger)


def do_val(cfg, epoch, model, dataloader, device, logger=None):
    model.eval()
    loss_cae_val = 0.0
    loss_traj_val = 0.0
    if cfg.method=='tprn_re_np': 
        if not "".__eq__(cfg.model.latent_dist): 
            loss_goal_val = 0.
            loss_kld_val = 0.
        if cfg.model.discrepancy: loss_disc_val = 0.
        # if cfg.model.best_of_many_back: loss_goal_back_val = 0.
    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader)):
            # For ETH_UCY dataset only
            if cfg.dataset.name in ['eth', 'hotel', 'univ', 'zara1', 'zara2', 'sdd']:
                input_x = batch['input_x_st']
                target_y = batch['target_y_st']
                input_imgs = batch['input_imgs']
                target_imgs = batch['target_imgs']
                if cfg.model.point_rel_enable:
                    neighbors_x = batch['neighbors_x_st']
                else:
                    neighbors_x = None
                # adjacency = restore(batch['neighbors_adjacency'])
                first_history_indices = batch['first_history_index']
                # if cfg.dataset.normalize == 'zero-one':
                input_x_frame = batch['input_x_frame']
                # else:
                    # input_x_frame = batch['input_x_st'][:,:,:cfg.model.motion.dec_output_dim]
            else:
                input_x = None
                # neighbors_st, adjacency, first_history_indices = None, None, None
            
            loss_dict, _ = model(input_x=input_x, input_imgs=input_imgs, input_x_frame=input_x_frame, target_y=target_y, target_imgs=target_imgs, neighbors_x=neighbors_x, first_history_indices=first_history_indices, mode = ModeKeys.EVAL)
            
            # compute loss
            if 'loss_cae' in loss_dict:
                loss_cae_val += loss_dict['loss_cae'].item()
            loss_traj_val += loss_dict['loss_traj'].item()
            if cfg.method=='tprn_re_np' and not "".__eq__(cfg.model.latent_dist): 
                loss_goal_val += loss_dict['loss_goal'].item()
                loss_kld_val += loss_dict['loss_kld'].item() 
                if cfg.model.discrepancy: loss_disc_val += loss_dict['loss_disc'].item() * cfg.loss.weight_disc
                # if cfg.model.best_of_many_back: loss_goal_back_val += loss_dict['loss_goal_back'].item()
    
    loss_traj_val /= (iters + 1)
    loss_cae_val /= (iters + 1)
    loss_val = loss_traj_val + loss_cae_val
    
    if cfg.method=='tprn_re_np' and not "".__eq__(cfg.model.latent_dist): 
        loss_val += (loss_goal_val + loss_kld_val) / (iters + 1) 
        if cfg.model.discrepancy: loss_val += loss_disc_val  / (iters + 1)
        if cfg.model.best_of_many_back:
            loss_val += loss_goal_back_val / (iters + 1) 
            info = "loss_val:{:.4f}, \
                loss_goal_val:{:.4f}, \
                loss_goal_back_val:{:.4f}, \
                loss_traj_val:{:.4f}, \
                loss_kld_val:{:.4f}, \
                loss_cae_val:{:.4f}".format(loss_val, loss_goal_val, loss_goal_back_val, loss_traj_val, loss_kld_val, loss_cae_val)
        else:
            info = "loss_val:{:.4f}, \
                loss_goal_val:{:.4f}, \
                loss_traj_val:{:.4f}, \
                loss_kld_val:{:.4f}, \
                loss_disc_val:{:.4f}, \
                loss_cae_val:{:.4f}".format(loss_val, loss_goal_val, loss_traj_val, loss_kld_val, loss_disc_val, loss_cae_val)
    else:
        info = "loss_val:{:.4f}, \
                loss_traj_val:{:.4f}, \
                loss_cae_val:{:.4f}".format(loss_val, loss_traj_val, loss_cae_val) 
    
    if hasattr(logger, 'log_values'):
        logger.info(info)
        logger.log_values({'loss_val':loss_val, 
                           'loss_traj_val':loss_traj_val, 
                           'loss_cae_val':loss_cae_val})#, step=epoch)
    else:
        print(info)
    return loss_val
        

def inference(cfg, epoch, model, dataloader, device, logger=None):
    model.eval()
    # evaluation metrics
    metrics = eval_metrics(cfg.test.metrics)
    # prediction tracker 
    analyst = PredAnalyst(cfg) if 'test' in cfg.stats_anl.subsets else None 
    if cfg.dataset.name in ['eth', 'hotel', 'univ', 'zara1', 'zara2', 'sdd']:
        viz = Visualizer(mode='plot')
    else:
        viz = Visualizer(mode='image')
    
    if cfg.test.eval_kde_nll:
        dataloader_params ={
            "batch_size": cfg.test.kde_batch_size,
            "shuffle": False,
            "num_workers": cfg.dataloader.num_workers,
            "collate_fn": dataloader.collate_fn,
            } ## follow the Trajectron++ KDE-based NLL evaluation, we generate 2000 predictions per input at each prediction timestep 
        kde_nll_dataloader = DataLoader(dataloader.dataset, **dataloader_params)
        metrics_kde_nll = inference_kde_nll(cfg, epoch, model, kde_nll_dataloader, device, logger)
        for k, v in metrics_kde_nll.items():
            metrics[k] = v
    
        return metrics
    
    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader)):
            X_global = batch['input_x']
            y_global = batch['target_y']
            # For ETH_UCY dataset only
            if cfg.dataset.name in ['eth', 'hotel', 'univ', 'zara1', 'zara2', 'sdd']:
                input_x = batch['input_x_st']
                target_y = batch['target_y_st']
                input_imgs = batch['input_imgs']
                target_imgs = batch['target_imgs']
                if cfg.model.point_rel_enable:
                    neighbors_x = batch['neighbors_x_st']
                else:
                    neighbors_x = None
                # adjacency = restore(batch['neighbors_adjacency'])
                first_history_indices = batch['first_history_index']
                if cfg.dataset.name != 'sdd':
                    homog_files = batch['homography_file'] # list
                else:
                    homog_files = [None]*len(input_imgs)
                frame_res = batch['frame_resolution'] # list of two tensors, width[0][N] and height[1][N]
                # if cfg.dataset.normalize == 'zero-one':
                input_x_frame = batch['input_x_frame']
                # else:
                #     input_x_frame = batch['input_x_st'][:,:,:cfg.model.motion.dec_output_dim]
            else:
                input_x = None
                # neighbors_st, adjacency, first_history_indices = None, None, None

            if cfg.model.convae.vis_dec:
                pred_imgs = model(input_x=input_x, input_imgs=input_imgs, input_x_frame=input_x_frame, target_y=target_y, first_history_indices=first_history_indices, mode=ModeKeys.PREDICT)
                save_dir = f'figures/future_densitymaps/{cfg.method}/{cfg.dataset.name}/batch{iters}'
                os.makedirs(save_dir, exist_ok=True)
                for i in range(len(input_imgs)):
                    for t in range(cfg.model.input_len):
                        save_image(input_imgs[i, t, ::], os.path.join(save_dir, f'input{i}_ts{t}.png'))
#                 for i in range(len(target_imgs)):
                    for t in range(cfg.model.pred_len):
                        save_image(target_imgs[i, t, ::], os.path.join(save_dir, f'target{i}_ts{t+cfg.model.input_len}.png'))
                        save_image(pred_imgs[i, t, ::], os.path.join(save_dir, f'recon{i}_ts{t+cfg.model.input_len}.png'))
                return
            else:
                loss_dict, pred_traj, best_idxs = model(input_x=input_x, input_imgs=input_imgs, input_x_frame=input_x_frame, target_y=target_y, neighbors_x=neighbors_x, first_history_indices=first_history_indices, mode=ModeKeys.PREDICT)

            if cfg.model.best_of_many:
                bok = 'best_of_many'
            elif cfg.model.z_mode:
                bok = 'z_mode'
            elif cfg.model.best_of_many_back:
                bok = 'best_of_many_back'

            if not cfg.dataset.frame_pred:
                # transfer back to global coordinates
                ret = post_process(cfg, X_global, y_global, pred_traj, frame_res=frame_res, homog_files=homog_files)
                X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal = ret
                ## need to visualize the reconstruction. visualize the prediction as well
                # if cfg.VISUALIZE and iters % max(int(len(dataloader)/5), 1) == 0:
                #     viz_results(viz, X_global, y_global, pred_traj, img_path, dist_goal, dist_traj, 
                #                 bbox_type=cfg.DATASET.BBOX_TYPE, normalized=False, logger=logger, name='pred_test')

                # Evaluate
                mode = 'bbox' if y_global.shape[-1] == 4 else 'point'
                eval_results = evaluate_multimodal(cfg, pred_traj, y_global, mode=mode, distribution=None, bbox_type=cfg.dataset.bbox_type, bok=bok, best_idxs=best_idxs) ## distribution is None
                # update metrics
                for name, value in eval_results.items():
                    if name in metrics:
                        metrics[name].update(value, y_global.shape[0]) # running average, when each sample is an object

            else: # frame-wise evaluation
                X_global = restore(X_global)
                y_global = restore(y_global)
                frame_res = restore(frame_res)
                nodeIDs = restore(batch['node_id'])
                timesteps = ((batch['timestep']+batch['timestep_min']+1)*10).tolist()
                assert len(pred_traj)==len(y_global)==len(nodeIDs)==len(batch['scene_name'])==len(batch['timestep']+batch['timestep_min'])==len(frame_res)
                for i in range(len(pred_traj)):
                    # transfer back to global coordinates
                    ret = post_process(cfg, torch.stack(X_global[i]), torch.stack(y_global[i]), pred_traj[i], frame_res=[[frame_res[i][0]]*len(pred_traj[i]), [frame_res[i][1]]*len(pred_traj[i])], homog_files=[homog_files[i]]*len(pred_traj[i]))
                    _, _y_global, pred_goal, _pred_traj, dist_traj, dist_goal = ret
                    # Evaluate
                    mode = 'bbox' if _y_global.shape[-1] == 4 else 'point'
                    best_idx = None if len(best_idxs)==0 else best_idxs[i].detach().to('cpu').numpy()
                    eval_results, best_idx = evaluate_multimodal(cfg, _pred_traj, _y_global, mode=mode, distribution=None, bbox_type=cfg.dataset.bbox_type, bok=bok, best_idx=best_idx, analyst=analyst) ## distribution is None
                    # update metrics
                    for name, value in eval_results.items():
                        if name in metrics:
                            metrics[name].update(value, _y_global.shape[0]) # running average, when each sample is an object
                    if analyst is not None:
                        analyst.nodeID_list.extend(nodeIDs[i])
                        # record the meta information including sceneID, frameID, pedID, pred_traj
                        analyst.sceneID_list.extend([batch['scene_name'][i],]*len(nodeIDs[i]))
                        analyst.frameID_list.extend([timesteps[i],]*len(nodeIDs[i]))

                        pred_traj_best = (pred_traj[i][np.arange(len(pred_traj[i])), :, best_idx, :] ) # n, 12,2 
                        # print(pred_traj_best.shape)
                        pred_traj_best[:,:,0] = pred_traj_best[:,:,0] * frame_res[i][0]
                        pred_traj_best[:,:,1] = pred_traj_best[:,:,1] * frame_res[i][1]
                        analyst.pred_state_list.extend(pred_traj_best.tolist())

                        gt_traj_best = torch.stack(y_global[i]) # n, 12,2 
                        # print(gt_traj_best.shape)
                        homog = np.loadtxt(batch['homography_file'][i])
                        y_uv = np.asarray([dataloader.dataset.to_image_frame(np.linalg.inv(homog), gt_traj) for gt_traj in gt_traj_best])
                        # import pdb; pdb.set_trace()
                        # gt_traj_best[:,:,0] = gt_traj_best[:,:,0] * frame_res[i][0]
                        # gt_traj_best[:,:,1] = gt_traj_best[:,:,1] * frame_res[i][1]
                        analyst.gt_state_list.extend(y_uv.tolist())

        if analyst is not None: # zip all the lists 
            pred_lists = [analyst.sceneID_list, analyst.frameID_list, analyst.nodeID_list, analyst.gt_state_list, analyst.pred_state_list, analyst.pred_metric_lists['ADE'], analyst.pred_metric_lists['FDE']]
            pred_cols = ['sceneID', 'frameID', 'nodeID', 'gt_state', 'pred_state', 'ADE', 'FDE'] # then sort the ADE/FDE
            all_list = list(zip(*pred_lists)) 
            df = pd.DataFrame(all_list, columns = pred_cols)
            pf.write_feather(df, os.path.join(cfg.ckpt_dir, cfg.dataset.name, f'seed{cfg.seed}', cfg.out_dir, f'predanl_{cfg.stats_anl.mode}_test.ftr'))
            dataFrame = pf.read_feather(os.path.join(cfg.ckpt_dir, cfg.dataset.name, f'seed{cfg.seed}', cfg.out_dir, f'predanl_{cfg.stats_anl.mode}_test.ftr'))

        for name, am in metrics.items():
            info = "Testing prediction {}:{}".format(name, str(np.around(am.avg, decimals=3)))
            if hasattr(logger, 'log_values'):
                logger.info(info)
            else:
                print(info)
        
        if hasattr(logger, 'log_values'):
            logger.log_values({k: v.avg for k, v in metrics.items()})
    
    # Evaluate Kernel Density Estimation Negative Log-likelihood
    if cfg.test.eval_kde_nll:
        dataloader_params ={
            "batch_size": cfg.test.kde_batch_size,
            "shuffle": False,
            "num_workers": cfg.dataloader.num_workers,
            "collate_fn": dataloader.collate_fn,
            } ## follow the Trajectron++ KDE-based NLL evaluation, we generate 2000 predictions per input at each prediction timestep 
        kde_nll_dataloader = DataLoader(dataloader.dataset, **dataloader_params)
        metrics_kde_nll = inference_kde_nll(cfg, epoch, model, kde_nll_dataloader, device, logger)
        for k, v in metrics_kde_nll.items():
            metrics[k] = v
    
    return metrics


def inference_kde_nll(cfg, epoch, model, dataloader, device, logger=None):
    model.eval()
    # evaluation metrics
    metrics = eval_metrics(names=['kde_nll', 'per_step_kde_nll'])

    num_samples = model.cfg.k
    model.cfg.k = cfg.test.kde_num_samples # 2000
 
    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader)):
            X_global = batch['input_x'] #.to(device)
            y_global = batch['target_y']
            # For ETH_UCY dataset only
            if cfg.dataset.name in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                input_x = batch['input_x_st'] #.to(device)
                target_y = batch['target_y_st']
                input_imgs = batch['input_imgs']
                if cfg.model.point_rel_enable:
                    neighbors_x = batch['neighbors_x_st']
                else:
                    neighbors_x = None
                # adjacency = restore(batch['neighbors_adjacency'])
                first_history_indices = batch['first_history_index']
                homog_files = batch['homography_file'] # list
                frame_res = batch['frame_resolution'] # list of two tensors, width[0][N] and height[1][N]
                input_x_frame = batch['input_x_frame']
            else:
                input_x = None
                # neighbors_st, adjacency, first_history_indices = None, None, None
            
            loss_dict, pred_traj, best_idxs = model(input_x=input_x, input_imgs=input_imgs, input_x_frame=input_x_frame, target_y=target_y, neighbors_x=neighbors_x, first_history_indices=first_history_indices, mode=ModeKeys.PREDICT)
            if not cfg.dataset.frame_pred:
                # transfer back to global coordinates
                ret = post_process(cfg, X_global, y_global, pred_traj, frame_res=frame_res, homog_files=homog_files)
                X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal = ret

                # Evaluate
                mode = 'bbox' if y_global.shape[-1] == 4 else 'point'
                eval_results = evaluate_multimodal(cfg, pred_traj, y_global, mode=mode, distribution=None, bbox_type=cfg.dataset.bbox_type, bok=bok, best_idxs=best_idxs) ## distribution is None
                # update metrics
                for name, value in eval_results.items():
                    if name in metrics:
                        metrics[name].update(value, y_global.shape[0]) # running average, when each sample is an object

            else: # frame-wise evaluation
                X_global = restore(X_global)
                y_global = restore(y_global)
                frame_res = restore(frame_res)
                assert len(pred_traj)==len(y_global)
                for i in range(len(pred_traj)):
                    # transfer back to global coordinates
                    ret = post_process(cfg, torch.stack(X_global[i]), torch.stack(y_global[i]), pred_traj[i], frame_res=[[frame_res[i][0]]*len(pred_traj[i]), [frame_res[i][1]]*len(pred_traj[i])], homog_files=[homog_files[i]]*len(pred_traj[i]))
                    _, _y_global, pred_goal, _pred_traj, dist_traj, dist_goal = ret

            # for i in range(len(pred_traj)):
                    KDE_NLL, KDE_NLL_PER_STEP = compute_kde_nll(_pred_traj, _y_global)
                    # running average
                    metrics['kde_nll'].update(KDE_NLL, _y_global.shape[0])
                    metrics['per_step_kde_nll'].update(KDE_NLL_PER_STEP, _y_global.shape[0])

        KDE_NLL = metrics['kde_nll'].avg
        KDE_NLL_PER_STEP = metrics['per_step_kde_nll'].avg

        # Evaluate
        Goal_NLL = KDE_NLL_PER_STEP[-1]
        nll_dict = {'KDE_NLL': KDE_NLL} if cfg.model.latent_dist == 'categorical' else {'KDE_NLL': KDE_NLL, 'Goal_NLL': Goal_NLL}
        info = "Testing prediction KDE_NLL:{:.4f}, per step NLL:{}".format(KDE_NLL, KDE_NLL_PER_STEP)
        if hasattr(logger, 'log_values'):
            logger.info(info)
        else:
            print(info)
        if hasattr(logger, 'log_values'):
            logger.log_values(nll_dict)

    # reset samples
    model.cfg.k = num_samples
    return metrics
