"""""""""
dataloader to load density maps of Stanford Drone Dataset
"""""""""
import os
import glob
import numpy as np
import torch.utils.data as data
from tqdm import tqdm
import h5py
from PIL import Image
import pandas as pd
from pyarrow import feather as pf
import torch


class DensitySDD(data.Dataset):
    def __init__(self, args, split, in_channel=1, transform=None, target_transform=None):
        """
        Args:
            root (str): Directory that contains splited datasets
            split (list or str): Type of split to load (e.g. 'train', 'val')
            in_channel (int): Number of input channel (e.g. 1 for grayscale data, 3 for RGB data)
            transform (class): Transform applied to the input
            target_transform (class): Transform applied to the target image of reconstruction
            cls (int): An inlier class
        """
        datalist_path = os.path.join(args.datalist_root, f'{split}.lst')
        self.vid_list = np.hstack(pd.read_csv(datalist_path, header=None).values).tolist()
        # self.data_dir =  #self.get_dset_path(args.dataset, split)
        # all_files = os.listdir(self.data_dir)
        # all_files = [os.path.join(self.data_dir, _path) for _path in all_files]

        self.dm_path_list = []
        for vid in self.vid_list:
            full_dm_root = os.path.join(args.img_path, 'cache_maps_fixed_kernel', os.path.splitext(vid)[0].replace('_','/'))        
            self.dm_path_list.extend(glob.glob(os.path.join(full_dm_root, '*.ftr')))

        self.transform = transform
        self.target_transform = target_transform
        self.in_channel = in_channel

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            img (tensor): Input image after transform
            target (tensor): Target image after transform for the reconstruction
            label (tensor): Class label for input image
        """
        map_path = self.dm_path_list[index]
        # load raw density maps
        img = np.vstack((pf.read_feather(map_path)['density_map'])[0][0])
        img = torch.from_numpy(np.expand_dims(img, axis=0)).float() 
#         # normalize to 0-1
#         density_map_norm = (density_map - np.min(density_map)) / (np.max(density_map) - np.min(density_map)+1e-10)

#         if self.in_channel == 1:
#             img = Image.fromarray(np.uint8(density_map_norm * 255), mode='L') 
#         elif self.in_channel == 3:
#             img = Image.fromarray(np.uint8(density_map_norm * 255), mode='RGB')
#         # target = img

#         if self.transform is not None:
#             img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return img, img

    def __len__(self):
        return len(self.dm_path_list)

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
    
    # def get_dset_path(self, dset_name, dset_type):
    #     _dir = os.getcwd()
    #     return os.path.join(_dir, 'eth_ucy_pixel', dset_name, dset_type)
    
if __name__ == '__main__':
    print()
    