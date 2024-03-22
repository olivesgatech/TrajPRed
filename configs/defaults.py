import os
from yacs.config import CfgNode as cn
"""contains default nested structure in `model` parameters"""
_c = cn()

_c.use_wandb = False
_c.project = ''
_c.ckpt_dir = ''
_c.out_dir = ''
_c.device = 'cuda'
_c.method = ''
_c.gpu = '1'
_c.seed = None
_c.visualize = False
_c.vis_attn = False
_c.print_interval = 20
# ------ model ---
_c.model = cn()
# global configs
_c.model.img_size = (256,256)  # resizing resolution of full frames 
_c.model.input_len = 15 # for 30 fps, 15 is 0.5 second
_c.model.pred_len = 45 # for 30 fps, 15 is 0.5 second
_c.model.maximum_history_length = 7
_c.model.use_traffic = False
_c.model.pred_intent = False
_c.model.num_pedintent_class = 3
_c.model.goal_hidden_size = 64
_c.model.latent_dim = 32
_c.model.best_of_many = False
_c.model.discrepancy = False
_c.model.best_of_many_back = False # evaluate based on the closest backward goal estimation
_c.model.z_mode = False
_c.model.k = 20
_c.model.latent_dist = ''
_c.model.latent_tensor = ''
_c.model.dec_with_z = False
_c.model.dec_backward_disable = False
# motion configs
_c.model.motion = cn()
_c.model.motion.global_input_dim = 4  # trajectory dimension 
_c.model.motion.enc_input_size = 256 ##trajectory embedding
_c.model.motion.enc_hidden_size = 256
_c.model.motion.dec_input_size = 256
_c.model.motion.dec_hidden_size = 256
_c.model.motion.dec_output_dim = 4
_c.model.motion.enc_concat_type = 'average'
# temporal enc/dec configs
_c.model.tempenc = ''
_c.model.tempdec = ''
_c.model.tempdec_hidden_size = 0
# rnn configs
_c.model.rnn_dropout_keep_prob = 1.
_c.model.mlp_dropout_keep_prob = 1.
# convae configs
_c.model.convae = cn()
_c.model.convae.image_size = 0
_c.model.convae.latent_dim = 0
_c.model.convae.ckpt_dir = '' 
_c.model.convae.out_dir = ''
_c.model.convae.test_epoch = -1
_c.model.convae.vis_dec = False
_c.model.convae.loss = ''
_c.model.convae.pretrain = True
_c.model.convae.load_pretrain_weights = True
# visual configs
_c.model.visual = cn()
_c.model.visual.crop_type='' 
_c.model.visual.pad_mode = 'pad_resize'
_c.model.visual.pad_size = 128
_c.model.visual.eratio=0.
_c.model.visual.use_attn = False  # whether to use pedestrian `look` attributes
_c.model.visual.use_agent_roi = False # whether to use rois
_c.model.visual.use_traffic_mask = False  # whether to use the union masks of traffic objects and target object
_c.model.visual.use_frame = False # whether to use full frames
_c.model.visual.use_scene_states = False # whether to use scene states there..
_c.model.visual.num_hs = 0
_c.model.visual.num_ws = 0
_c.model.visual.backbone = ''
_c.model.visual.backbone_dim = 0  # the feat. dim after the feature extraction (and before final pooling)
_c.model.visual.backbone_pretrained = True
_c.model.visual.backbone_freeze = False
_c.model.visual.rand_relation = False
_c.model.visual.head = ''  # contrastive head network
_c.model.visual.head_feat_dim = 0
_c.model.visual.loss_temp = 0.07  # temperature in loss
# graph configs
_c.model.graph = cn()
_c.model.graph.adj_type = '' # the type of adjacency matrix
_c.model.graph.collect_adj = False
_c.model.graph.n_lyrs = 1 # number of graph conv layers
_c.model.graph.diff_lyr_weight = False
_c.model.graph.conv_dim = 256
_c.model.region_rel_combine = ''
_c.model.region_rel_disable = False
_c.model.region_rel_mask = False
_c.model.region_rel_backbone = 'LSTM'
_c.model.point_rel_enable = False
_c.model.edge_state_combine_method = "sum"
# ----- dataset -----
_c.dataset = cn()
_c.dataset.name = ''
_c.dataset.root = ''
_c.dataset.image_path = ''
_c.dataset.density_kernel_size = ''
_c.dataset.density_kernel_sigma = 0
_c.dataset.track_overlap = 0.5
_c.dataset.excl_ocln = False
_c.dataset.fps = 30.
_c.dataset.ts_intent = 0.8 # time step of short-term intent (seconds after the current time)
_c.dataset.mode = '2d'
_c.dataset.bbox_type = 'cxcywh' # bbox is in cxcywh format
_c.dataset.normalize = 'zero-one' # normalize to 0-1
_c.dataset.excl_unseen_node = False
_c.dataset.perturb_input = False
_c.dataset.frame_pred = False
_c.dataset.min_bbox = [0,0,0,0] # the min of cxcywh or x1x2y1y2
_c.dataset.max_bbox = [1920, 1080, 1920, 1080] # the max of cxcywh or x1x2y1y2
_c.dataset.augment = False
_c.dataset.ethucy_config = ''
_c.dataset.sdd_config = ''
_c.dataset.trajectory_path = ''
_c.dataset.trajectory_annot_root = ''
_c.dataset.cacher = cn()
_c.dataset.cacher.path = ''
# ---- dataloader -----
_c.dataloader = cn()
_c.dataloader.num_workers = 4
# ------ test ----- 
_c.test = cn()
_c.test.batch_size = 128
_c.test.inference = False
_c.test.epoch = 0
_c.test.metrics = ['ADE', 'FDE']
_c.test.eval_kde_nll = False
_c.test.kde_num_samples = 0
_c.test.kde_batch_size = 0
# ------ statistics analysis ----- 
_c.stats_anl = cn()
_c.stats_anl.subsets = []
_c.stats_anl.metrics = []
_c.stats_anl.mode = '3d'
# ------ solver ----
_c.solver = cn()
_c.solver.max_epoch = 30
_c.solver.batch_size = 512
_c.solver.optimizer = ''
_c.solver.lr = 0.001
_c.solver.lr_decay_rate = 0.
_c.solver.patience = 10
_c.solver.lr_steps = []
_c.solver.scheduler = ''
_c.solver.cosine = False
_c.solver.momentum = 0.9
_c.solver.gamma = 0.999
_c.solver.weight_decay = 1e-4
# ------ loss weights ----
_c.loss = cn()
_c.loss.weight_traj = 1.
_c.loss.weight_disc = 0.01
_c.loss.weight_intent = 1.
