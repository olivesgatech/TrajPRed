""" 
generating density maps for eth-ucy trajectories
"""
import os
import glob
import argparse
import numpy as np
from PIL import Image
import h5py
import scipy
import scipy.ndimage
from scipy import spatial
from itertools import islice
import pickle
from tqdm import tqdm
from sortedcontainers import SortedDict
import matplotlib.pyplot as plt
import pdb

class DensityMaps:
    def __init__(self, args):
        super(DensityMaps, self).__init__()
        self.args = args
        # load Homography  
        homog = (np.loadtxt(args.hmg_file)) if os.path.exists(args.hmg_file) else None
        print(f"homography of {args.hmg_file.split('/')[-2]}:\n{homog}")
        
        if args.regen_data_pixel:  # generate trajectories in the pixel space
            self.generate_data_pixel(homog)
        self.kernels_dict=None
        self.distances_dict=None
    
    def generate_density_maps(self, min_sigma = 2, method = 1, map_out_folder = 'maps_adaptive_kernel/'):
        ## `min_sigma` can be set 0 
        data = self.read_txt_file(self.args.data_pixel_path)  # all trajectory data
        frames = np.unique(data[:, 0]).tolist()

        curr_map_out_folder =  f'{self.args.img_path}_{map_out_folder}' 
        if not os.path.isdir(curr_map_out_folder):
            print('creating ' + curr_map_out_folder)
            os.makedirs(curr_map_out_folder)
        for frame in tqdm(frames):
            # data_folder, img_path = full_img_path.split('images')
            # mat_path = full_img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_')
            full_img_path = os.path.join(self.args.img_path,f'{int(frame):05d}.png')
            # load img and map
            try: img = Image.open(full_img_path)
            except FileNotFoundError: pass
            width, height = img.size
            gt_points = self.get_gt_dots(data[frame == data[:, 0], 2:], height, width)

            distances = self.distances_dict[full_img_path]
            density_map = self.gaussian_filter_density(gt_points, height, width, distances, min_sigma=min_sigma, method=method)

            gt_out_path = curr_map_out_folder + f'{int(frame):05d}.h5'

            save_computed_density(density_map, gt_out_path)
        
    def gaussian_filter_density(self, non_zero_points, map_h, map_w, distances=None, min_sigma=2, method=1, const_sigma=15):
        """
        Fast gaussian filter implementation : using precomputed distances and kernels
        """
        gt_count = non_zero_points.shape[0]
        density_map = np.zeros((map_h, map_w), dtype=np.float32)
        non_zero_points = non_zero_points.round().astype(int)
        for i in range(gt_count):
            point_y, point_x = non_zero_points[i]
            sigma = compute_sigma(gt_count, distances[i], min_sigma=min_sigma, method=method, fixed_sigma=const_sigma)
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

    def compute_distances(self, out_dist_path='distances_dict.pkl', img_path='./', n_neighbors = 4, leafsize=1024):
        distances_dict = dict()
        data = self.read_txt_file(self.args.data_pixel_path)  # all trajectory data
        frames = np.unique(data[:, 0]).tolist()
        # full_img_pathes = glob.glob() # glob.glob(f'{img_path}*/*/images/*.jpg')

        for frame in tqdm(frames):
            # mat_path = full_img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_')
            full_img_path = os.path.join(self.args.img_path,f'{int(frame):05d}.png')
            try: img = plt.imread(full_img_path)
            except FileNotFoundError: pass
 
            non_zero_points = self.get_gt_dots(data[frame == data[:, 0], 2:], *img.shape[0:2])

            tree = spatial.KDTree(non_zero_points.copy(), leafsize=leafsize)  # build kdtree
            distances, _ = tree.query(non_zero_points, k=n_neighbors)  # query kdtree

            distances_dict[full_img_path] = distances

        print(f'Distances computed for {len(frames)} frames. Saving them to {out_dist_path}')

        with open(out_dist_path, 'wb') as f:
            pickle.dump(distances_dict, f)

    def generate_data_pixel(self, homog):
        # load ground-truth trajectories in world space
        data = self.read_txt_file(self.args.data_world_path)  # all trajectory data
        frames = np.unique(data[:, 0]).tolist()
        frame_data = []
        frame_data_pixel = []
        for frame in frames:
            frame_data.append(data[frame == data[:, 0], :])
        # project to pixel space
        for idx, frame in tqdm(enumerate(frames)): 
            _data_uv = self.to_image_frame(np.linalg.inv(homog),  frame_data[idx][:,2:])
            print(f'idx: {idx}, frame: {frame}')
            frame_data_pixel.append(np.hstack((frame_data[idx][:,:2], _data_uv)))
        frame_data_pixel = np.vstack(frame_data_pixel)
        # save to disk
        os.makedirs('/'.join(self.args.data_pixel_path.split('/')[:-1]),exist_ok=True)
        # with open(os.path.join(self.args.data_pixel_path)) as f:
        np.savetxt(self.args.data_pixel_path, frame_data_pixel, delimiter='\t')
        
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

    def get_gt_dots(self, gt, img_height, img_width):
        """
        ** cliping is needed to prevent going out of the array
        """
        # mat = io.loadmat(mat_path)
        # gt = mat["image_info"][0,0][0,0][0].astype(np.float32).round().astype(int)
        gt[:,0] = gt[:,0].clip(0, img_width - 1)
        gt[:,1] = gt[:,1].clip(0, img_height - 1)
        return gt
    
    def load_distances_dict(self, precomputed_distances_path):
        with open(precomputed_distances_path, 'rb') as f:
            self.distances_dict = pickle.load(f)
            
    def load_kernels_dict(self, precomputed_kernels_path):
        with open(precomputed_kernels_path, 'rb') as f:
            self.kernels_dict = pickle.load(f)
            self.kernels_dict = SortedDict(self.kernels_dict)

def generate_gaussian_kernels(out_kernels_path='gaussian_kernels.pkl', round_decimals = 3, sigma_threshold = 4, sigma_min=0, sigma_max=20, num_sigmas=801):
    """
    Computing gaussian filter kernel for sigmas in linspace(sigma_min, sigma_max, num_sigmas) and saving 
    them to dict.     
    """
    kernels_dict = dict()
    sigma_space = np.linspace(sigma_min, sigma_max, num_sigmas)
    for sigma in tqdm(sigma_space):
        sigma = np.round(sigma, decimals=round_decimals) 
        kernel_size = np.ceil(sigma*sigma_threshold).astype(np.int)

        img_shape  = (kernel_size*2+1, kernel_size*2+1)
        img_center = (img_shape[0]//2, img_shape[1]//2)

        arr = np.zeros(img_shape)
        arr[img_center] = 1

        arr = scipy.ndimage.filters.gaussian_filter(arr, sigma, mode='constant') 
        kernel = arr / arr.sum()
        kernels_dict[sigma] = kernel
        
    print(f'Computed {len(sigma_space)} gaussian kernels. Saving them to {out_kernels_path}')

    with open(out_kernels_path, 'wb') as f:
        pickle.dump(kernels_dict, f)

        
def compute_sigma(gt_count, distance=None, min_sigma=1, method=1, fixed_sigma=15):
    """
    Compute sigma for gaussian kernel with different methods :
    * method = 1 : sigma = (sum of distance to 3 nearest neighbors) / 10
    * method = 2 : sigma = distance to nearest neighbor
    * method = 3 : sigma = fixed value
    ** if sigma lower than threshold 'min_sigma', then 'min_sigma' will be used
    ** in case of one point on the image sigma = 'fixed_sigma'
    """    
    if gt_count > 1 and distance is not None:
        if   method == 1:
            sigma = np.mean(distance[1:4])*0.1
        elif method == 2:
            sigma = distance[1]
        elif method == 3:
            sigma = fixed_sigma
    else:
        sigma = fixed_sigma
    if sigma < min_sigma:
        sigma = min_sigma
    return sigma


def find_closest_key(sorted_dict, key):
    """
    Find closest key in sorted_dict to 'key'
    """
    keys = list(islice(sorted_dict.irange(minimum=key), 1))
    keys.extend(islice(sorted_dict.irange(maximum=key, reverse=True), 1))
    return min(keys, key=lambda k: abs(key - k))

def save_computed_density(density_map, out_path):
    """
    Save density map to h5py format
    """
    with h5py.File(out_path, 'w') as hf:
        hf['density'] = density_map
        
def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--hmg_file", type=str, default='')
    parser.add_argument("--img_path", type=str, default='')
    parser.add_argument("--data_world_path", type=str, default='')
    parser.add_argument("--regen_data_pixel", default=False, action='store_true', help='generate trajectories in the pixel space')
    parser.add_argument("--data_pixel_path", type=str, default='')
    # parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()    

if __name__ == '__main__':
    args = parse_args()
    for k,v in vars(args).items():
        print(k,": ",v)
    
    dm = DensityMaps(args)
    
    #step0: generate trajectoreis in pixel space via homography
    # setp1: generate density maps
     
    # generate gaussian kernels
    precomputed_kernels_path = './eth_ucy_pixel/gaussian_kernels.pkl'

    # uncomment to generate and save dict with kernel sizes
    # generate_gaussian_kernels(precomputed_kernels_path, round_decimals=3, sigma_threshold=4, sigma_min=0, sigma_max=20, num_sigmas=801)
    
    dm.load_kernels_dict(precomputed_kernels_path)

    # different dictionaries to store different scenes
    precomputed_distances_path = args.data_pixel_path.replace('.txt', '_distances_dict.pkl')

    # uncomment to generate and save dict with distances 
    # dm.compute_distances(out_dist_path=precomputed_distances_path, img_path=args.img_path)

    dm.load_distances_dict(precomputed_distances_path)

    dm.generate_density_maps(method = 3, map_out_folder = 'maps_fixed_kernel/')
