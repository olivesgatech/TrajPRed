import os
import sys
from .preprocessing import get_node_timestep_data
sys.path.append(os.path.realpath('./datasets/trajectory'))
from environment import derivative_of
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms as tvtfs
import pickle
import dill
import json
import h5py
from tqdm import tqdm
import pandas as pd
from sortedcontainers import SortedDict
from datasets.density.density_gen import compute_sigma, find_closest_key
from pyarrow import feather as pf


# HOMOGRAPHY_FILES = {
#     'biwi_eth': 'ynet_additional_files/data/eth_ucy/eth_h.txt',
#     'biwi_hotel': 'ynet_additional_files/data/eth_ucy/hotel_h.txt',
#     'crowds_zara01': 'ynet_additional_files/data/eth_ucy/zara1_H.txt',
#     'crowds_zara02': 'ynet_additional_files/data/eth_ucy/zara2_H.txt',
#     'crowds_zara03': 'ynet_additional_files/data/eth_ucy/zara3_H.txt',
#     'students001': 'ynet_additional_files/data/eth_ucy/students001_H.txt',
#     'students003': 'ynet_additional_files/data/eth_ucy/students003_H.txt',
#     'uni_examples': 'ynet_additional_files/data/eth_ucy/uni_examples_H.txt',
# }

class SDDDataset(data.Dataset):
    def __init__(self, cfg, split, transforms=None):
        self.cfg = cfg
        self.split = split

        conf_json = open(cfg.dataset.sdd_config, 'r')
        hyperparams = json.load(conf_json)
        
        hyperparams['minimum_history_length'] = self.cfg.model.input_len-1 if self.split == 'test' else 1
        hyperparams['maximum_history_length'] = self.cfg.model.input_len-1
        
        # hyperparams['minimum_history_length'] = cfg.model.MIN_HIST_LEN #1 # different from trajectron++, we don't use short histories.
        hyperparams['state'] = {'Pedestrian':{'position':['x','y'], 'velocity':['x','y'], 'acceleration':['x','y']}}
        hyperparams['pred_state'] = {'Pedestrian':{'position':['x','y']}}
# with open('../misc/Trajectron-plus-plus/experiments/sdd/test.pkl', 'rb') as f:

        with open(os.path.join(cfg.dataset.trajectory_path, cfg.dataset.name, f'{split}.pkl'), 'rb') as f:
            train_env = dill.load(f, encoding='latin1')

        for i,_ in enumerate(train_env.NodeType):
            if str(train_env.NodeType[i]) == 'Pedestrian':
                node_type=train_env.NodeType[i]
                break
        train_env.standardization['Pedestrian']['velocity']['x']['std'] = train_env.standardization['Pedestrian']['position']['x']['std']*2
        train_env.standardization['Pedestrian']['velocity']['y']['std'] = train_env.standardization['Pedestrian']['position']['y']['std']*2
        train_env.standardization['Pedestrian']['acceleration']['x']['std'] = train_env.standardization['Pedestrian']['position']['x']['std']
        train_env.standardization['Pedestrian']['acceleration']['y']['std'] = train_env.standardization['Pedestrian']['position']['y']['std']
        # train_env.attention_radius[(node_type, node_type)] = 3.0 #1.0
        augment = False
        if split=='train':
            min_history_timesteps = 1
            augment = True if self.cfg.dataset.augment else False
        else:
            min_history_timesteps = 7
        self.dataset = NodeTypeDataset(train_env, 
                                        node_type, 
                                        hyperparams['state'],
                                        hyperparams['pred_state'],
                                        scene_freq_mult=hyperparams['scene_freq_mult_train'],
                                        node_freq_mult=hyperparams['node_freq_mult_train'],
                                        hyperparams=hyperparams, 
                                        frame_pred=cfg.dataset.frame_pred,
                                        augment=augment, 
                                        min_history_timesteps=min_history_timesteps,
                                        min_future_timesteps=hyperparams['prediction_horizon'],
                                        return_robot=False)
        # get some statistics on the dataset.
#         all_obs_distance, all_obs_vel = [], []
#         all_pred_distance, all_pred_vel = [], []
#         for d in tqdm(self.dataset):
#             distance = torch.norm(d[1][-1, :2] - d[1][0, :2])
#             all_obs_distance.append(distance)
#             all_obs_vel.append(d[1][:, 2:4])

#             distance = torch.norm(d[2][-1] - d[2][0])
#             all_pred_distance.append(distance)
#             all_pred_vel.append((d[2][1:] - d[2][:-1])/0.4) ## 1/FPS=0.4hz

#         all_obs_vel = torch.cat(all_obs_vel, dim=0).norm(dim=1)
#         all_obs_distance = torch.tensor(all_obs_distance)
#         all_pred_vel = torch.cat(all_pred_vel, dim=0).norm(dim=1)
#         all_pred_distance = torch.tensor(all_pred_distance)
        
#         print("obs dist mu/sigma: {:.2f}/{:.2f}, obs vel mu/sigma: {:.2f}/{:.2f}, pred dist mu/sigma: {:.2f}/{:.2f}, pred vel mu/sigma: {:.2f}/{:.2f}".format(\
#                 all_obs_distance.mean(), all_obs_distance.std(), all_obs_vel.mean(), all_obs_vel.std(),
                # all_pred_distance.mean(), all_pred_distance.std(), all_pred_vel.mean(), all_pred_vel.std()))
        self.dt = 0.4
        if transforms is None:
            self.transforms = tvtfs.Compose([tvtfs.Resize((cfg.model.convae.image_size, cfg.model.convae.image_size)),
                                             tvtfs.ToTensor()])
        else:
            self.transforms = transforms

        if self.cfg.dataset.excl_unseen_node:
            self.traj_frame_data = {}
            traj_framedata_dir = os.path.join('eth_ucy_pixel', cfg.dataset.name, split)
            all_files = [os.path.join(traj_framedata_dir, _path) for _path in os.listdir(traj_framedata_dir) if _path.endswith('.txt')]
            for path in all_files:
                print(f'load trajectory (frame) data from: {path}')
                self.traj_frame_data[path] = pd.DataFrame(self.read_txt_file(path), columns = ['frameID','nodeID','u','v']) 
            with open('./eth_ucy_pixel/gaussian_kernels.pkl', 'rb') as f:
                self.kernels_dict = pickle.load(f)
                self.kernels_dict = SortedDict(self.kernels_dict)
        
        if self.cfg.dataset.perturb_input and self.cfg.test.inference:
            raise NotImplementedError
            self.traj_world_data = {}
            traj_worlddata_dir = os.path.join('../sgan/datasets', cfg.dataset.name, split)
            all_files = [os.path.join(traj_worlddata_dir, _path) for _path in os.listdir(traj_worlddata_dir) if _path.endswith('.txt')]
            for path in all_files:
                print(f'load trajectory (frame) data from: {path}')
                self.traj_world_data[path] = pd.DataFrame(self.read_txt_file(path), columns = ['frameID','nodeID','u','v']) 

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        if self.cfg.dataset.frame_pred: # frame-wise prediction
            ret = {}
            # define a set of lists
            ret['input_x'] = []
            ret['input_x_st'] = []
            ret['input_x_frame'] = []
            ret['target_y'] = []
            ret['target_y_st'] = []
            ret['first_history_index'] = []
            ret['node_id'] = []
            fhi_min = float("inf") 
            if self.cfg.model.point_rel_enable:
                ret['neighbors_x_st'] = []
 
            nodes_timestep_data = self.dataset.__getitem__(index)
            for node_timestep_data in nodes_timestep_data:
                _ret = self._getnode_(node_timestep_data)
                if _ret['first_history_index'] < fhi_min:
                    fhi_min = _ret['first_history_index']
                    ret['input_imgs'] = _ret['input_imgs']
                ret['input_x'].append(_ret['input_x'])
                ret['input_x_st'].append(_ret['input_x_st'])
                ret['input_x_frame'].append(_ret['input_x_frame'])
                ret['target_y'].append(_ret['target_y'])
                ret['target_y_st'].append(_ret['target_y_st'])
                ret['first_history_index'].append(_ret['first_history_index'])
                ret['node_id'].append(_ret['node_id'])
                if self.cfg.model.point_rel_enable:
                    ret['neighbors_x_st'].append(_ret['neighbors_x'])
            # the following values are consistent in the current scene and timestep
            ret['target_imgs'] = _ret['target_imgs']
            ret['scene_name'] = _ret['scene_name']
            # ret['homography_file'] = _ret['homography_file']
            ret['frame_resolution'] = _ret['frame_resolution']
            ret['timestep_min'] = _ret['timestep_min']
            ret['timestep'] = _ret['timestep']
            assert len(ret['input_x']) == len(ret['input_x_st']) == len(ret['input_x_frame']) == len(ret['target_y']) == len(ret['target_y_st']) == len(ret['first_history_index']) == len(ret['node_id'])
            return ret 
        else:
            return self._getnode_(self.dataset.__getitem__(index)) # 

    def _getnode_(self, node_timestep_data):
        first_history_index, x_t, y_t, x_st_t, y_st_t, neighbors_data, neighbors_data_st, neighbors_lower_upper, neighbors_future, \
            neighbors_edge_value, robot_traj_st_t, map_tuple, scene_name, timestep, timestep_min, node_id = node_timestep_data
        ret = {}
        ret['first_history_index'] = first_history_index
        ret['input_x'] = x_t
        ret['input_x_st'] = x_st_t
        ret['target_y'] = y_t
        ret['target_y_st'] = y_st_t
        ret['cur_image_file'] = ''
        ret['pred_resolution'] = torch.ones_like(y_t)
        ret['neighbors_x'] = neighbors_data
        ret['neighbors_x_st'] = neighbors_data_st
        ret['neighbors_lower_upper'] = neighbors_lower_upper
        ret['neighbors_target_y'] = neighbors_future
        ret['neighbors_adjacency'] = neighbors_edge_value
        ret['scene_name'] = scene_name
        ret['timestep'] = timestep
        ret['timestep_min'] = timestep_min
        ret['node_id'] = int(node_id)
        
        # load density maps
        ret['input_imgs'], ret['target_imgs'], height, width = self.load_density_maps(scene_name, timestep, timestep_min, first_history_index) # input_density_maps, target_density_maps
        # print(ret['input_imgs'].shape, ret['target_imgs'].shape) # torch.Size([8, 1, 160, 160]) torch.Size([12, 1, 160, 160])
        # _scene_name = scene_name.split('/')[-1].split('.')[0].replace('_train', '').replace('_val', '')
        # load Homography  
        # homog = (np.loadtxt(HOMOGRAPHY_FILES[_scene_name])) if os.path.exists(HOMOGRAPHY_FILES[_scene_name]) else None
        # print(ret['input_x']) # this is 0-1 normalized now
        # print(ret['input_x_st']) # this is useless now
        if self.cfg.dataset.perturb_input and self.cfg.test.inference:
            x_uv_t = x_t[:,:2]+0.02*np.random.randn(x_t[:,:2].shape[0], x_t[:,:2].shape[1])
        else:
            x_uv_t = x_t[:,:2] 
        y_uv_t = y_t[:,:2]
        if self.cfg.model.point_rel_enable: ## transform neighboring nodes
            for i, neighbor in enumerate(neighbors_data[self.dataset.edge_types[0]]): # list of neighbors...# x_uv_t = self.to_image_frame(np.linalg.inv(homog), x_t[:,:2])
                neighbor_uv = neighbor[:,:2].numpy()
                if self.cfg.dataset.normalize == 'zero-one':  #Normalize agent states
                    # normalize to 0-1 
                    pass
                    # neighbor_uv[:, 0], neighbor_uv[:, 1] = neighbor_uv[:, 0]/width, neighbor_uv[:, 1]/height
                else:
                    # normalize to same dim as density maps
                    neighbor_uv[:, 0], neighbor_uv[:, 1] = neighbor_uv[:, 0]*self.cfg.model.convae.image_size, neighbor_uv[:, 1]*self.cfg.model.convae.image_size
                # then derivative
                vx, vy, ax, ay = self.calc_vel_accel(neighbor_uv[:, 0], neighbor_uv[:, 1])
                neighbors_data[self.dataset.edge_types[0]][i] = torch.Tensor(np.hstack((neighbor_uv, vx.reshape(-1,1), vy.reshape(-1,1), ax.reshape(-1,1), ay.reshape(-1,1))))
            ret['neighbors_x'] = neighbors_data
        if self.cfg.dataset.normalize == 'zero-one':  #Normalize agent states
            # normalize to 0-1 
            pass
            # x_uv_t[:, 0], x_uv_t[:, 1] = x_uv_t[:, 0]/width, x_uv_t[:, 1]/height
            # y_uv_t[:, 0], y_uv_t[:, 1] = y_uv_t[:, 0]/width, y_uv_t[:, 1]/height
        else:
            # normalize to same dim as density maps
            x_uv_t[:, 0], x_uv_t[:, 1] = x_uv_t[:, 0]*self.cfg.model.convae.image_size, x_uv_t[:, 1]*self.cfg.model.convae.image_size
            y_uv_t[:, 0], y_uv_t[:, 1] = y_uv_t[:, 0]*self.cfg.model.convae.image_size, y_uv_t[:, 1]*self.cfg.model.convae.image_size
        
        if torch.is_tensor(x_uv_t):
            x_uv_t = x_uv_t.numpy()
        if torch.is_tensor(y_uv_t):
            y_uv_t = y_uv_t.numpy()
        
        # then derivative
        vx, vy, ax, ay = self.calc_vel_accel(x_uv_t[:, 0], x_uv_t[:, 1])
        
        ret['input_x_st'] = torch.from_numpy(np.hstack((x_uv_t, vx.reshape(-1,1), vy.reshape(-1,1), ax.reshape(-1,1), ay.reshape(-1,1)))).float()
        ret['target_y_st'] = torch.from_numpy(y_uv_t).float()
        if self.cfg.dataset.normalize == 'zero-one':
            ret['input_x_frame'] = torch.from_numpy(x_uv_t*self.cfg.model.convae.image_size).float()
        else:
            ret['input_x_frame'] = torch.from_numpy(x_uv_t).float()
        # for post-processing
        ret['frame_resolution'] = [width, height]
        # ret['homography_file'] = HOMOGRAPHY_FILES[_scene_name]
        if self.cfg.dataset.excl_unseen_node: # create target maps by using only agents appeared in hist. sequence.
            traj_data = self.traj_frame_data[os.path.join('eth_ucy_pixel/', ret['scene_name'])]
            ret['target_imgs'] = self.create_target_maps(traj_data,timestep,timestep_min,height,width)
        
        # print(x_uv_t) # shape (8,2)
        if self.cfg.dataset.perturb_input and self.cfg.test.inference:
            # load world data to perturb inputs.
            # traj_data = self.traj_world_data[os.path.join('../sgan/datasets', ret['scene_name'])]
            ret['input_imgs'] = self.create_input_maps(timestep,timestep_min,height,width, scene=ret['scene_name'], perturb_input=self.cfg.dataset.perturb_input)
        
        return ret
    
    
    def calc_vel_accel(self,x,y):
        vx = derivative_of(x, self.dt)
        vy = derivative_of(y, self.dt)
        ax = derivative_of(vx, self.dt)
        ay = derivative_of(vy, self.dt)
        
        return vx, vy, ax, ay
    
    def load_density_maps(self,path,timestep,timestep_min,fhi):
        full_dm_root = os.path.join(self.cfg.dataset.image_path, 'maps_fixed_kernel', '/'.join(path.split('/')[-2:]))
#         print(timestep) # 2
#         print(timestep_min) # 0
#         print(fhi)# 5

        # if self.split=='test':
        #     _scene = path.split('/')[-1].split('_')[-1].split('.')[-2]
        # else:
        #     _scene = path.split('/')[-1].split('_')[-2]

        # full_dm_root = glob.glob(f'{self.cfg.dataset.image_path}/maps_{self.cfg.dataset.density_kernel_size}_kernel_sigma{self.cfg.dataset.density_kernel_sigma}/*{_scene}*maps_{self.cfg.dataset.density_kernel_size}_kernel_sigma{self.cfg.dataset.density_kernel_sigma}')[0]
        height,width = np.vstack( pf.read_feather(glob.glob(os.path.join(full_dm_root, '*'))[0])['density_map'][0] ).shape
        # height,width = np.asarray(h5py.File(glob.glob(os.path.join(full_dm_root, '*.h5'))[0], 'r')['density']).shape
        # load input density
        input_density_maps = []
        for i, frame in enumerate(range((timestep_min+timestep-self.cfg.model.input_len+1)*1, (timestep_min+timestep+1)*1, 1)):
            # print(i, frame)
            if i<fhi: # the current dummy frame is not present in the scene
                input_density_maps.append(torch.zeros(1,self.cfg.model.convae.image_size,self.cfg.model.convae.image_size))
            else:
                # input_density_maps.append(torch.from_numpy(np.asarray(h5py.File(os.path.join(full_dm_root,f'{int(frame):05d}.h5'), 'r')['density'])))
                input_density_maps.append(self._load_transformed_dms(full_dm_root, frame))
        # load target density
        if self.cfg.dataset.excl_unseen_node:
            target_density_maps = (torch.tensor([]),)
        else:
            target_density_maps = []
            for i, frame in enumerate(range((timestep_min+timestep+1)*1, (timestep_min+timestep+self.cfg.model.pred_len+1)*1, 1)):
                # print(i, frame)
                # target_density_maps.append(torch.from_numpy(np.asarray(h5py.File(os.path.join(full_dm_root,f'{int(frame):05d}.h5'), 'r')['density'])))
                target_density_maps.append(self._load_transformed_dms(full_dm_root, frame))
        
        input_density_maps, target_density_maps = torch.stack(input_density_maps, dim=0), torch.stack(target_density_maps, dim=0) 
        
        return input_density_maps, target_density_maps, height, width
    
    def _load_transformed_dms(self, dm_root, frame, in_channel=1):
        # add the opt to load from saved tensors.
        cache_root = dm_root.replace(f'{self.cfg.dataset.image_path}/', f'{self.cfg.dataset.image_path}/cache_')
        cache_path = os.path.join(cache_root,f'{int(frame)}.ftr')

        if not os.path.exists(cache_path): 
            os.makedirs(cache_root, exist_ok=True)
            # load raw density maps
            try:
                density_map = np.vstack(pf.read_feather(os.path.join(dm_root, f'{int(frame)}.ftr'))['density_map'][0])
            except: #FileNotFoundError:
                return torch.zeros(1,self.cfg.model.convae.image_size,self.cfg.model.convae.image_size)

            # normalize to 0-1
            density_map_norm = (density_map - np.min(density_map)) / (np.max(density_map) - np.min(density_map))

            if in_channel == 1:
                img = Image.fromarray(np.uint8(density_map_norm * 255), mode='L') 
            elif in_channel == 3:
                img = Image.fromarray(np.uint8(density_map_norm * 255), mode='RGB')

            if self.transforms is not None:
                img = self.transforms(img)
            # print(img.shape) # torch.Size([1, H, W])
            pf.write_feather(pd.DataFrame(list(zip([img.numpy().tolist(),])), columns=['density_map']), cache_path)
            # with open(cache_path, 'wb') as fid:
            #     pickle.dump(img, fid, pickle.HIGHEST_PROTOCOL)
        else:
            # with open(cache_path, 'rb') as fid:
            #     img = pickle.load(fid)  #  tensor
            img = np.vstack((pf.read_feather(cache_path)['density_map'])[0][0])
            img = torch.from_numpy(np.expand_dims(img, axis=0)).float() 

        return img
    
    def to_image_frame(self, homog, loc):
        """Given H^-1 and world coordinates, returns (u, v) in image coordinates."""
        locHomogenous = np.hstack((loc, np.ones((loc.shape[0], 1))))
        loc_tr = np.transpose(locHomogenous)
        loc_tr = np.matmul(homog, loc_tr)  # to camera frame
        locXYZ = np.transpose(loc_tr / loc_tr[2])  # to pixels (from millimeters)
        return locXYZ[:, :2]
    
    def read_txt_file(self, _path, delim='\t'):
        """read raw trajectory data"""
        data = []
        if delim == 'tab':
            delim = '\t'
        elif delim == 'space':
            delim = ' '
        with open(_path, 'r') as f:
            for line in f:
                line = line.strip().split(delim)
                line = [float(i) for i in line]
                data.append(line)
        return np.asarray(data)
    
    def create_target_maps(self,timestep,timestep_min,height,width,in_channel=1):
        target_maps = []
        input_frames = list(range((timestep_min+timestep-self.cfg.model.input_len+1)*1, (timestep_min+timestep+1)*1, 1))
        input_nodeIDs = np.unique(traj_data.query('frameID in @input_frames').nodeID)
        for i, frame in enumerate(range((timestep_min+timestep+1)*1, (timestep_min+timestep+self.cfg.model.pred_len+1)*1, 1)):
            _traj_target_df = traj_data.query('frameID==@frame and nodeID in @input_nodeIDs')
#             if len(_traj_target_df)==0: raise ValueError('A very specific bad thing happened.')
            gt_points = self.get_gt_dots(_traj_target_df[['u','v']].values, height, width)
            density_map = self.gaussian_filter_density(gt_points, height, width,)
            # normalize to 0-1
            density_map_norm = (density_map - np.min(density_map)) / (np.max(density_map) - np.min(density_map))
            if in_channel == 1:
                img = Image.fromarray(np.uint8(density_map_norm * 255), mode='L') 
            elif in_channel == 3:
                img = Image.fromarray(np.uint8(density_map_norm * 255), mode='RGB')
            if self.transforms is not None:
                img = self.transforms(img)
            target_maps.append(img)
        return torch.stack(target_maps, dim=0)

    def create_input_maps(self,traj_data,timestep,timestep_min,height,width,scene,perturb_input,in_channel=1):
        target_maps = []
        input_frames = list(range((timestep_min+timestep-self.cfg.model.input_len+1)*1, (timestep_min+timestep+1)*1, 1))
        annot_path = os.path.join(self.cfg.dataset.trajectory_annot_root, '/'.join(scene.split('/')[-2:]), 'annotations.txt')
        annot = pd.read_csv(annot_path, sep=' ', names=['TrackID', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label'])
        for i, frame in enumerate(input_frames):
            _traj_target_df = annot.query('lost!=1 & frame==@frame')
            # _traj_target_df = traj_data.query('frameID==@frame')
            if len(_traj_target_df)==0:
                target_maps.append(torch.zeros(1,self.cfg.model.convae.image_size,self.cfg.model.convae.image_size))
                continue
            pos_x = (_traj_target_df[['xmin', 'xmax']].values).mean(1)
            pos_y = (_traj_target_df[['ymin', 'ymax']].values).mean(1)
            _traj_target = np.hstack((pos_x.reshape(-1,1), pos_y.reshape(-1,1)))
            ## transform from world to frame.
            if perturb_input:
                x_uv_t = _traj_target + 0.02*height*np.random.randn(_traj_target.shape[0], _traj_target.shape[1])

            else:
                x_uv_t = _traj_target
            gt_points = self.get_gt_dots(x_uv_t, height, width)
            density_map = self.gaussian_filter_density(gt_points, height, width,)
            # normalize to 0-1
            density_map_norm = (density_map - np.min(density_map)) / (np.max(density_map) - np.min(density_map))
            if in_channel == 1:
                img = Image.fromarray(np.uint8(density_map_norm * 255), mode='L') 
            elif in_channel == 3:
                img = Image.fromarray(np.uint8(density_map_norm * 255), mode='RGB')
            if self.transforms is not None:
                img = self.transforms(img)
            target_maps.append(img)
        return torch.stack(target_maps, dim=0)
    
    def gaussian_filter_density(self, non_zero_points, map_h, map_w, distances=None, min_sigma=2, method=1):
        """
        Fast gaussian filter implementation : using precomputed distances and kernels
        """
        gt_count = non_zero_points.shape[0]
        density_map = np.zeros((map_h, map_w), dtype=np.float32)
        non_zero_points = non_zero_points.round().astype(int)
        for i in range(gt_count):
            point_y, point_x = non_zero_points[i]
            if self.cfg.dataset.density_kernel_size == 'fixed':
                sigma = self.cfg.dataset.density_kernel_sigma
            else:
                sigma = compute_sigma(gt_count, distances[i], min_sigma=min_sigma, method=method, fixed_sigma=self.cfg.dataset.density_kernel_sigma)
            closest_sigma = find_closest_key(self.kernels_dict, sigma)
            kernel = self.kernels_dict[closest_sigma]
            full_kernel_size = kernel.shape[0]
            kernel_size = full_kernel_size // 2

            min_img_x = max(0, point_x-kernel_size)
            min_img_y = max(0, point_y-kernel_size)
            max_img_x = min(point_x+kernel_size+1, map_h - 1)
            max_img_y = min(point_y+kernel_size+1, map_w - 1)

            kernel_x_min = kernel_size - point_x if point_x <= kernel_size else 0
            kernel_y_min = kernel_size - point_y if point_y <= kernel_size else 0
            kernel_x_max = kernel_x_min + max_img_x - min_img_x
            kernel_y_max = kernel_y_min + max_img_y - min_img_y

            density_map[min_img_x:max_img_x, min_img_y:max_img_y] += kernel[kernel_x_min:kernel_x_max, kernel_y_min:kernel_y_max]
        return density_map

    def get_gt_dots(self, gt, img_height, img_width):
        """
        ** cliping is needed to prevent going out of the array
        """
        # mat = io.loadmat(mat_path)
        # gt = mat["image_info"][0,0][0,0][0].astype(np.float32).round().astype(int)
        gt[:,0] = gt[:,0].clip(0, img_width - 1)
        gt[:,1] = gt[:,1].clip(0, img_height - 1)
        return gt

class NodeTypeDataset(data.Dataset):
    '''
    from Trajectron++: https://github.com/StanfordASL/Trajectron-plus-plus
    '''
    def __init__(self, env, node_type, state, pred_state, node_freq_mult,
                 scene_freq_mult, hyperparams, frame_pred, augment=False, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']
        self.frame_pred = frame_pred
        self.augment = augment

        self.node_type = node_type
        self.index = self.index_env(node_freq_mult, scene_freq_mult, **kwargs)
        self.len = len(self.index)
        self.edge_types = [edge_type for edge_type in env.get_edge_types() if edge_type[0] is node_type]
    def index_env(self, node_freq_mult, scene_freq_mult, **kwargs):
        index = list()
        
        for scene in self.env.scenes:
            present_node_dict = scene.present_nodes(np.arange(0, scene.timesteps), type=self.node_type, **kwargs)
            for t, nodes in present_node_dict.items():
                if not self.frame_pred:
                    for node in nodes:
                        index += [(scene, t, node)] *\
                                 (scene.frequency_multiplier if scene_freq_mult else 1) *\
                                 (node.frequency_multiplier if node_freq_mult else 1)
                else:
                    index += [(scene, t, nodes)]

        return index

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        if not self.frame_pred:
            (scene, t, node) = self.index[i]

            if self.augment:
                scene = scene.augment()
                node = scene.get_node_by_id(node.id)
            return get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
                                          self.edge_types, self.max_ht, self.max_ft, self.hyperparams)
        else:
            nodes_timestep_data = []
            (scene, t, nodes) = self.index[i]
            for node in nodes:
                if self.augment:
                    scene = scene.augment()
                    node = scene.get_node_by_id(node.id)
                nodes_timestep_data.append(get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
                                                                  self.edge_types, self.max_ht, self.max_ft, self.hyperparams))
            return nodes_timestep_data

if __name__=='__main__':
    dataset = SDDDataset(cfg, split)
