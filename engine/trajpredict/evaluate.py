import numpy as np
from pyarrow import feather as pf
from sklearn.metrics import pairwise_distances
from scipy.stats import gaussian_kde
# import pdb

def evaluate_multimodal(cfg, pred_traj, target_traj, mode='bbox', distribution=None, bbox_type='x1y1x2y2', bok=None, best_idx=None, analyst=None):
    """
    assuming that `pred_traj` is transformed to global coordinate
    bok: 'best_of_many' | 'best_of_many_back'
    """
    if len(pred_traj.shape)==4:
        K = pred_traj.shape[2]  # if no K-mode prediction
        tiled_target_traj = np.tile(target_traj[:, :, None, :], (1, 1, K, 1))
    elif len(pred_traj.shape)==len(target_traj.shape)==3:
        pred_traj=np.tile(pred_traj[:, :, None, :], (1, 1, 1, 1))
        tiled_target_traj = np.tile(target_traj[:, :, None, :], (1, 1, 1, 1))
    elif len(pred_traj.shape)==len(target_traj.shape)==2:
        pred_traj=np.tile(pred_traj[None, :, None, :], (1, 1, 1, 1))
        tiled_target_traj = np.tile(target_traj[None, :, None, :], (1, 1, 1, 1))    
    if mode == 'bbox':
        eval_results = evaluate_bbox_traj(pred_traj, tiled_target_traj, bbox_type=bbox_type)
    elif mode == 'point':
        eval_results = {}
        traj_ADE = np.linalg.norm(pred_traj - tiled_target_traj, axis=-1).mean(1)
        traj_FDE = np.linalg.norm(pred_traj - tiled_target_traj, axis=-1)[:, -1]
        if bok == 'best_of_many':
            eval_results['ADE'] = np.min(traj_ADE, axis=1).mean()
            eval_results['FDE'] = np.min(traj_FDE, axis=1).mean()
            eval_results['per_step_displacement_error'] = np.linalg.norm(pred_traj - tiled_target_traj, axis=-1).min(axis=2).mean(axis=0)
            if analyst is not None:
                if best_idx is None: best_idx = np.argmin(traj_ADE, axis=1)
                analyst.pred_metric_lists['ADE'].extend( np.min(traj_ADE, axis=1) )
                analyst.pred_metric_lists['FDE'].extend( np.min(traj_FDE, axis=1) )
        elif bok == 'z_mode':
            eval_results['ADE'] = traj_ADE[:, 0].mean()
            eval_results['FDE'] = traj_FDE[:, 0].mean()
            eval_results['per_step_displacement_error'] = np.linalg.norm(pred_traj - tiled_target_traj, axis=-1)[:, :, 0].mean(axis=0)
        elif bok == 'best_of_many_back' and best_idx is not None:
            eval_results['ADE'] = traj_ADE[range(len(best_idx)), best_idx].mean()
            eval_results['FDE'] = traj_FDE[range(len(best_idx)), best_idx].mean()
            eval_results['per_step_displacement_error'] = np.linalg.norm(pred_traj - tiled_target_traj, axis=-1)[range(len(best_idx)), :, best_idx].mean(axis=0)
        # eval_results = evaluate_point_traj(pred_traj[range(len(best_idx)), best_idx], target_traj)
        for metric in cfg.test.metrics:
            if metric in ['ADE','FDE']: continue
            if metric.startswith('ADE'):
                T = float(metric.split('ADE')[-1].replace('(','').replace(')','').replace('s',''))
                ts = int(np.floor(T/(1/cfg.dataset.fps)+0.5))
                _traj_ADE = np.linalg.norm(pred_traj[:, :ts, :, :] - tiled_target_traj[:, :ts, :, :], axis=-1).mean(1)
                if bok == 'best_of_many':
                    eval_results[metric] = np.min(_traj_ADE, axis=1).mean()
                elif bok == 'z_mode':
                    eval_results[metric] = _traj_ADE[:, 0].mean()
                elif bok == 'best_of_many_back' and best_idx is not None:
                    eval_results[metric] = _traj_ADE[range(len(best_idx)), best_idx].mean()
            elif metric.startswith('FDE'):
                T = float(metric.split('FDE')[-1].replace('(','').replace(')','').replace('s',''))
                ts = int(np.floor(T/(1/cfg.dataset.fps)+0.5))
                _traj_FDE = np.linalg.norm(pred_traj[:, :ts, :, :] - tiled_target_traj[:, :ts, :, :], axis=-1)[:, -1]
                if bok == 'best_of_many':
                    eval_results[metric] = np.min(_traj_FDE, axis=1).mean()
                elif bok == 'z_mode':
                    eval_results[metric] = _traj_FDE[:, 0].mean()
                elif bok == 'best_of_many_back' and best_idx is not None:
                    eval_results[metric] = _traj_FDE[range(len(best_idx)), best_idx].mean()
    else:
        raise NameError('Wrong mode')
    return eval_results, best_idx


def compute_kde_nll(predicted_trajs, gt_traj):
    """
    calculate sample negative log-likelihood per-timestep
    predicted_trajs: (batch, T, K, dim)
    gt_traj: (batch, T, dim)
    """
    assert gt_traj.shape[1]==predicted_trajs.shape[1]
    kde_ll = 0.
    log_pdf_lower_bound = -20
    num_timesteps = gt_traj.shape[1]
    num_batches = predicted_trajs.shape[0]
    kde_ll_per_step = np.zeros(num_timesteps)
    for batch_num in range(num_batches):
        for timestep in range(num_timesteps):
            try:
                kde = gaussian_kde(predicted_trajs[batch_num, timestep, :, ].T)
                pdf = np.clip(kde.logpdf(gt_traj[batch_num, timestep].T), a_min=log_pdf_lower_bound, a_max=None)[0]
                kde_ll += pdf / (num_timesteps * num_batches)
                kde_ll_per_step[timestep] += pdf / num_batches
            except np.linalg.LinAlgError:
                kde_ll = np.nan
                kde_ll_per_step[timestep] = np.nan

    return -kde_ll, -kde_ll_per_step
