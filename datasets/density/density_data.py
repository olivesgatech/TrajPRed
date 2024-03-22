"""""""""
dataloader to load density maps
"""""""""
import os
import glob
import numpy as np
import torch.utils.data as data
from tqdm import tqdm
import h5py
from PIL import Image
import pdb

class DensityDataset(data.Dataset):
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
        self.data_dir = self.get_dset_path(args.dataset, split)
        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]

        self.dm_path_list = []
        for path in all_files:
            if not path.endswith('.txt'): continue
            data = self.read_txt_file(path)  # all trajectory data for the current scene
            frames = np.unique(data[:, 0]).tolist()
            if split=='test' or split=='val':
                _scene = path.split('/')[-1].split('_')[-1].split('.')[-2] if split=='test' else path.split('/')[-1].split('_')[-2]
                full_dm_root = glob.glob(f'{args.img_path}/maps_{args.kernel_size}_kernel_sigma{args.const_sigma}/*{_scene}*maps_{args.kernel_size}_kernel_sigma{args.const_sigma}')[0]
                print(f'finding density maps from: {full_dm_root}')
                for frame in tqdm(frames):
                    self.dm_path_list.append(os.path.join(full_dm_root,f'{int(frame):05d}.h5'))
            else:
                _scene = path.split('/')[-1].split('_')[-2]
                full_dm_root_list = glob.glob(f'{args.img_path}/maps_{args.kernel_size}_kernel_sigma{args.const_sigma}/*{_scene}*maps*{args.kernel_size}_kernel_sigma{args.const_sigma}')
                for full_dm_root in full_dm_root_list:
                    print(f'finding density maps from: {full_dm_root}')
                    for frame in tqdm(frames):
                        self.dm_path_list.append(os.path.join(full_dm_root,f'{int(frame):05d}.h5'))


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
        with h5py.File(map_path, 'r') as hf:
            density_map = np.asarray( hf['density'] )
        # normalize to 0-1
        density_map_norm = (density_map - np.min(density_map)) / (np.max(density_map) - np.min(density_map)+1e-10)

        if self.in_channel == 1:
            img = Image.fromarray(np.uint8(density_map_norm * 255), mode='L') 
        elif self.in_channel == 3:
            img = Image.fromarray(np.uint8(density_map_norm * 255), mode='RGB')
        # target = img

        if self.transform is not None:
            img = self.transform(img)

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
    
    def get_dset_path(self, dset_name, dset_type):
        _dir = os.getcwd()
        return os.path.join(_dir, 'eth_ucy_pixel', dset_name, dset_type)
    
if __name__ == '__main__':
    print()
    