"""the script to augment the density data"""
import os
import argparse
import glob
import h5py
import numpy as np
import imutils
from tqdm import tqdm
import pdb


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--img_path", type=str, default='')
    parser.add_argument("--angle", type=int,)
    return parser.parse_args()    

def main():
    args = parse_args()
    print(args.angle)
    map_out_folder = 'maps_fixed_kernel/'
    # load density map
    curr_map_out_folder =  f'{args.img_path}_{map_out_folder}' 
    print(f'Loading density maps from: {curr_map_out_folder}')
    
    rotate_map_out_folder = f'{args.img_path}_maps_rotate{args.angle}_fixed_kernel'
    if not os.path.isdir(rotate_map_out_folder):
        print('creating ' + rotate_map_out_folder)
        os.makedirs(rotate_map_out_folder)
    # original density map files
    dm_path_list = (glob.glob(curr_map_out_folder+'*.h5'))
    for dm_path in tqdm(dm_path_list):
        with h5py.File(dm_path, 'r') as hf:
            density_map = np.asarray( hf['density'] )
        # rotate the density map
        density_map_rotated = imutils.rotate(density_map, args.angle)
        
        out_path = (os.path.join(rotate_map_out_folder, dm_path.split('/')[-1]))
        with h5py.File(out_path, 'w') as hf:
            hf['density'] = density_map_rotated

if __name__ == '__main__': 
     main()
