import dill
import collections.abc 
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader
from datasets.density.density_data import DensityDataset
from datasets.density.density_data_sdd import DensitySDD
from datasets.trajectory.ETH_UCY import ETHUCYDataset
from datasets.trajectory.SDD import SDDDataset
import torch

_DATA_LAYERS = {
    'eth': ETHUCYDataset,
    'hotel': ETHUCYDataset,
    'univ': ETHUCYDataset,
    'zara1': ETHUCYDataset,
    'zara2': ETHUCYDataset,
    'sdd': SDDDataset,
 }

def make_dataset(cfg, split):
    try:
        data_layer = _DATA_LAYERS[cfg.dataset.name]
    except:
        raise NameError("Unknown dataset:{}".format(cfg.dataset.name))
    
    return data_layer(cfg, split)

def make_dataloader(cfg, split='train',logger=None):
    if split == 'test':
        batch_size = cfg.test.batch_size
    else:
        batch_size = cfg.solver.batch_size
    dataloader_params ={
            "batch_size": batch_size,
            "shuffle":split == 'train',
            "num_workers": cfg.dataloader.num_workers,
            "collate_fn": collate_dict,
            "pin_memory":True
            }
    
    dataset = make_dataset(cfg, split)
    # if len(dataset)%cfg.solver.batch_size == 1: dataloader_params['drop_last'] = True
    dataloader = DataLoader(dataset, **dataloader_params)
    if hasattr(logger, 'info'):
        logger.info("{} dataloader: {}".format(split, len(dataloader)))
    else:
        print("{} dataloader: {}".format(split, len(dataloader)))
    return dataloader

def collate_dict(batch):
    '''
    batch: a list of dict
    '''
    if len(batch) == 0:
        return batch
    elem = batch[0]
    collate_batch = {}
    all_keys = list(elem.keys())
    for key in all_keys:
        # e.g., key == 'bbox' or 'neighbors_st' or so
        if elem[key] is None:
            collate_batch[key] = None
        elif isinstance(elem[key], list):
            node_list = [b[key] for b in batch]
            collate_batch[key] = dill.dumps(node_list) if torch.utils.data.get_worker_info() else node_list
        elif isinstance(elem[key], collections.abc.Mapping):
            # We have to dill the neighbors structures. Otherwise each tensor is put into
            # shared memory separately -> slow, file pointer overhead
            # we only do this in multiprocessing
            neighbor_dict = {sub_key: [b[key][sub_key] for b in batch] for sub_key in elem[key]}
            collate_batch[key] = dill.dumps(neighbor_dict) if torch.utils.data.get_worker_info() else neighbor_dict
        else:
            collate_batch[key] = default_collate([b[key] for b in batch])
    return collate_batch
