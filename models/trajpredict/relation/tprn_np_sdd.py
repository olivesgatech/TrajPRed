import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import numpy as np 
import argparse
from configs import cfg
from .base import BaseModel
from models.trajpredict.model_utils import run_lstm_on_variable_length_seqs, ModeKeys
from models.trajpredict.loss import cvae_loss, bom_traj_loss, cvae_z_mode_loss, cae_loss
from models.trajpredict.components import *
from models.receptive_field import receptive_field, receptive_field_for_unit
from datasets.trajectory.preprocessing import restore
from matplotlib import path
import itertools
import copy


class TrajPRed_NP_SDD(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = copy.deepcopy(cfg)
        self.region_rel_disable = cfg.region_rel_disable
        import pickle
        with open(os.path.expanduser('~/Trajectron-plus-plus/experiments/pedestrians/sdd_train_coords.pkl'), 'rb') as f:
            sdd_coords = pickle.load(f)

        self.state = dict()
        for c in sdd_coords.keys():
            self.state[c] = {"position": ["x", "y"], "velocity": ["x", "y"], "acceleration": ["x", "y"]}
        # self.state = {'Pedestrian': {"position": ["x", "y"], "velocity": ["x", "y"], "acceleration": ["x", "y"]}}
        # self.rand_relation = cfg.visual.rand_relation # for debugging    
        receptive_field_dict = receptive_field(self.cae, (1, cfg.convae.image_size, cfg.convae.image_size), device="cpu")
        self.rf_dict = dict()
        self.rf_center = torch.zeros(10,10,2)
        for row in range(10):
            for col in range(10): 
                xs, ys = receptive_field_for_unit(receptive_field_dict, "5", (col,row))
                self.rf_dict[path.Path([(xs[0],ys[0]), (xs[0], ys[1]), (xs[1], ys[1]), (xs[1], ys[0])])] = [row,col]
        
                self.rf_center[row,col,:] = (torch.tensor([xs[0],ys[0]]) + torch.tensor([xs[1], ys[1]])) /2

        if not cfg.point_rel_enable:
            self.temp_rel_enc = nn.LSTM(input_size=self.temp_enc.hidden_size+2, hidden_size=self.temp_enc.hidden_size, batch_first=True)
        else:
            state_length = int(np.sum([len(entity_dims) for entity_dims in self.state['Pedestrian'].values()]))
            neighbor_state_length = int(
                np.sum([len(entity_dims) for entity_dims in self.state['Pedestrian'].values()]))
            self.edge_encoder = nn.LSTM(input_size=state_length + neighbor_state_length, hidden_size=cfg.motion.enc_hidden_size, batch_first=True)
            # Chose additive attention because of https://arxiv.org/pdf/1703.03906.pdf
            # We calculate an attention context vector using the encoded edges as the "encoder"
            # (that we attend _over_)
            # and the node history encoder representation as the "decoder state" (that we attend _on_).
            self.edge_influence_encoder = AdditiveAttention(encoder_hidden_state_dim=self.edge_encoder.hidden_size,
                                                            decoder_hidden_state_dim=cfg.motion.enc_hidden_size)

            # self.eie_output_dims = self.hyperparams['enc_rnn_dim_edge_influence']
        # if cfg.tempdec == 'rnn': 
        #     self.temp_dec = nn.GRUCell(input_size=128+self.temp_enc.hidden_size+cfg.convae.latent_dim, hidden_size=cfg.tempdec_hidden_size)
        # for future encoding
        self.node_future_encoder_h = nn.Linear(self.cfg.motion.global_input_dim, 32)
        self.gt_goal_encoder = nn.GRU(input_size=self.cfg.motion.dec_output_dim,
                                        hidden_size=32,
                                        bidirectional=True,
                                        batch_first=True)

        # prior network
        self.p_z_x = nn.Sequential(nn.Linear(cfg.motion.enc_hidden_size, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, self.cfg.latent_dim*2))
        # recognition network (posterior)
        self.q_z_xy = nn.Sequential(nn.Linear(cfg.motion.enc_hidden_size+self.cfg.goal_hidden_size, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, self.cfg.latent_dim*2))

        # goal predictor
        self.goal_decoder = nn.Sequential(nn.Linear(cfg.motion.enc_hidden_size+self.cfg.latent_dim, 128),
                                            nn.ReLU(),
                                            nn.Linear(128, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, self.cfg.motion.dec_output_dim))

        if self.cfg.best_of_many_back:
            self.traj_enc_back = nn.LSTM(input_size=self.cfg.motion.dec_output_dim, hidden_size=cfg.motion.enc_hidden_size, batch_first=True)
            self.goal_decoder_back = nn.Sequential(nn.Linear(cfg.motion.enc_hidden_size, 128),
                                            nn.ReLU(),
                                            nn.Linear(128, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, self.cfg.motion.dec_output_dim))
        #  add bidirectional predictor
        del self.fut_dec_embed
        del self.fut_decoder  
        
        self.dec_init_hidden_size = cfg.motion.enc_hidden_size+self.temp_enc.hidden_size + self.cfg.latent_dim if self.cfg.dec_with_z else cfg.motion.enc_hidden_size+self.temp_enc.hidden_size
        if self.cfg.dec_backward_disable:
            self.enc_h_to_forward_h = nn.Sequential(nn.Linear(self.dec_init_hidden_size, self.cfg.motion.dec_hidden_size*2),
                                                nn.ReLU())
            self.traj_dec_input_forward = nn.Sequential(nn.Linear(self.cfg.motion.dec_hidden_size*2, self.cfg.motion.dec_input_size*2),
                                                    nn.ReLU())
            self.traj_dec_forward = nn.GRUCell(input_size=self.cfg.motion.dec_input_size*2, hidden_size=self.cfg.motion.dec_hidden_size*2) 
        else:
            self.enc_h_to_forward_h = nn.Sequential(nn.Linear(self.dec_init_hidden_size, self.cfg.motion.dec_hidden_size),
                                                nn.ReLU())
            self.traj_dec_input_forward = nn.Sequential(nn.Linear(self.cfg.motion.dec_hidden_size, self.cfg.motion.dec_input_size),
                                                    nn.ReLU())
            self.traj_dec_forward = nn.GRUCell(input_size=self.cfg.motion.dec_input_size, hidden_size=self.cfg.motion.dec_hidden_size) 
        
            self.enc_h_to_back_h = nn.Sequential(nn.Linear(self.dec_init_hidden_size, self.cfg.motion.dec_hidden_size),
                                            nn.ReLU())
            self.traj_dec_input_backward = nn.Sequential(nn.Linear(self.cfg.motion.dec_output_dim, self.cfg.motion.dec_input_size),
                                                    nn.ReLU())
            self.traj_dec_backward = nn.GRUCell(input_size=self.cfg.motion.dec_input_size, hidden_size=self.cfg.motion.dec_hidden_size)

        self.traj_output = nn.Linear(self.cfg.motion.dec_hidden_size * 2,
                                     self.cfg.motion.dec_output_dim)

    # note that we will have different training/inference method    
    def forward(self, input_x, 
                input_imgs,
                input_x_frame = None,
                target_y = None, 
                target_imgs = None,
                neighbors_x=None, 
                cur_pos = None, 
                first_history_indices = None,
                mode = None):
        """

        :param input_x: Input tensor including the state for each agent over time [bs, t, state].
        :param input_imgs: Tensor of density information. [bs, t, channels, h, w]
        :param target_y: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param target_imgs: Label tensor of density information. [bs, t, channels, h, w]
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :return:
        """
        loss_dict = {}
        # step1: pass history density maps through cae to obtain `hist_cae_z`
        hist_cae_z = []
        for t in range(input_imgs.shape[1]):
            hist_cae_z.append(self.cae.encode(input_imgs[:, t, ::].to(self.device))) 
        # step2: temporal filtering to obtain Spatiotemporal historical latent representations of density    
        N, _, H, W = hist_cae_z[0].shape
        if self.cfg.tempenc == 'rnn':
            hist_cae_z = torch.stack(hist_cae_z, dim=1) # torch.Size([N, input_len, convae_dim, H, W])
            # region-wise temporal encoding
            spatemp_feats = torch.zeros(N,self.input_len,self.temp_enc.hidden_size,H,W).to(self.device)
            for row in range(H):
                for col in range(W):
                    output, (_, _) = self.temp_enc(hist_cae_z[:, :, :, row, col])
                    output = F.dropout(output,
                                       p=1. - self.cfg.rnn_dropout_keep_prob,
                                       training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]
                    spatemp_feats[:, :, :, row, col] = output

        elif self.cfg.tempenc == 'conv':        
            hist_cae_z = torch.stack(hist_cae_z, dim=2)
            spatemp_feats = self.temp_enc(hist_cae_z).squeeze(2)  # torch.Size([N, C, H, W])

        if target_imgs is not None:
            # train and val
            if self.cfg.convae.loss == 'bce_hist':  # recon. hist. data
                loss_cae_z = 0
                for t in range(self.input_len):
                    loss_cae_z += F.binary_cross_entropy(self.cae.decode(hist_cae_z[:, t, ::]), input_imgs[:, t, ::].to(self.device))  # [bs, channels, h, w]
                loss_dict['loss_cae'] = loss_cae_z
            elif self.cfg.convae.loss == 'bce_fut': # recon. fut. data
                predict_imgs = []
                for t in range(self.pred_len):
                    predict_imgs.append(self.cae.decode(fut_cae_z_hat[:, :, t, :, :]))
                predict_imgs = torch.stack(predict_imgs, dim=1) 
                loss_cae_z = 0
                for t in range(self.pred_len):
                    loss_cae_z += F.binary_cross_entropy(predict_imgs[:, t, :, :, :], target_imgs[:, t, :, :, :].to(self.device))
                loss_dict['loss_cae'] = loss_cae_z
            elif self.cfg.convae.loss == 'mse':
                # pass future density maps through cae to obtain `fut_cae_z`
                fut_cae_z = []
                for t in range(self.pred_len):
                    fut_cae_z.append(self.cae.encode(target_imgs[:, t, ::].to(self.device)))
                fut_cae_z = torch.stack(fut_cae_z, dim=2)  # torch.Size([N, C, pred_len, H, W])
                # compute deviation loss between z and z_hat
                loss_cae_z = cae_loss(fut_cae_z_hat, fut_cae_z)
                loss_dict['loss_cae'] = loss_cae_z
            elif self.cfg.convae.loss == 'null': 
                loss_dict['loss_cae'] = torch.Tensor([0]).to(self.device)
        else:
            # inference
            if self.cfg.convae.vis_dec: # visualize the predicted density
                predict_imgs = []
                for t in range(self.pred_len):
                    predict_imgs.append(self.cae.decode(fut_cae_z_hat[:, :, t, :, :]))
                return torch.stack(predict_imgs, dim=1) 
        # step5: incorporating relation representations into forecasting agent future states.
        if not self.cfg.frame_pred:
            pass
        else:  # frame-wise prediction
            ## define 1) loss_goal 2) loss_traj 3) loss_kld
            if not "".__eq__(self.cfg.latent_dist): 
                loss_dict['loss_goal'] = 0
                loss_dict['loss_kld'] = 0
            # if self.cfg.best_of_many_back: loss_dict['loss_goal_back'] = 0
            loss_dict['loss_traj'] = 0

            if self.cfg.discrepancy: loss_dict['loss_disc'] = 0
            pred_y = []
            if mode == ModeKeys.PREDICT: best_idxs = []
            # the following lists are all of length N(batch-size)
            input_x = restore(input_x)
            input_x_frame = restore(input_x_frame)
            target_y = restore(target_y)
            first_history_indices = restore(first_history_indices)
            if self.cfg.point_rel_enable and neighbors_x is not None:
                neighbors_x = restore(neighbors_x)

            assert len(input_x) == N
            for i in range(len(input_x)): # iterate every sub-sequence
                n = len(input_x[i])   # the number of nodes in this sub-sequence
                if not "".__eq__(self.cfg.latent_dist):  # 'gaussian' | 'categorical'
                    loss_goal, loss_traj_y, loss_kld, _pred_y, loss_disc =   self.forward_seq(spatemp_feats[i].unsqueeze(0).repeat(n, 1, 1, 1, 1), # duplicate rel_cae_z[i] to number of nodes.
                                                          torch.stack(input_x[i]).to(self.device),  # torch.Size([n, input_len, global_input_dim])
                                                          torch.stack(input_x_frame[i]),   # torch.Size([n, input_len, dec_output_dim])
                                                          torch.stack(target_y[i]).to(self.device),  # torch.Size([n, pred_len, dec_output_dim])
                                                          neighbors_x = neighbors_x[i] if neighbors_x is not None else None,
                                                          first_history_indices = torch.tensor(first_history_indices[i], dtype=torch.int64),
                                                          mode = mode)
                    if self.cfg.discrepancy: loss_dict['loss_disc'] += loss_disc
                # elif self.cfg.best_of_many_back:
                #     loss_goal, loss_goal_back, loss_traj_y, loss_kld, _pred_y, _best_idx =   self.forward_seq(spatemp_feats[i].unsqueeze(0).repeat(n, 1, 1, 1, 1), # duplicate rel_cae_z[i] to number of nodes.
                #                                           torch.stack(input_x[i]).to(self.device),  # torch.Size([n, input_len, global_input_dim])
                #                                           torch.stack(input_x_frame[i]),   # torch.Size([n, input_len, dec_output_dim])
                #                                           torch.stack(target_y[i]).to(self.device),  # torch.Size([n, pred_len, dec_output_dim])
                #                                           neighbors_x = neighbors_x[i] if neighbors_x is not None else None,
                #                                           first_history_indices = torch.tensor(first_history_indices[i], dtype=torch.int64),
                #                                           mode = mode)
                #     loss_dict['loss_goal_back'] += loss_goal_back
                #     if mode == ModeKeys.PREDICT: best_idxs.append(_best_idx)
                    loss_dict['loss_kld'] += loss_kld
                    loss_dict['loss_goal'] += loss_goal

                else: # detereministic prediction
                    loss_traj_y, _pred_y =   self.forward_seq(spatemp_feats[i].unsqueeze(0).repeat(n, 1, 1, 1, 1), # duplicate rel_cae_z[i] to number of nodes.
                                                          torch.stack(input_x[i]).to(self.device),  # torch.Size([n, input_len, global_input_dim])
                                                          torch.stack(input_x_frame[i]),   # torch.Size([n, input_len, dec_output_dim])
                                                          torch.stack(target_y[i]).to(self.device),  # torch.Size([n, pred_len, dec_output_dim])
                                                          neighbors_x = neighbors_x[i] if neighbors_x is not None else None,
                                                          first_history_indices = torch.tensor(first_history_indices[i], dtype=torch.int64),
                                                          mode = mode)
                    
                loss_dict['loss_traj'] += loss_traj_y

                pred_y.append(_pred_y)

            if not "".__eq__(self.cfg.latent_dist):     
                loss_dict['loss_goal'] /= N
                loss_dict['loss_kld'] /= N
            # if self.cfg.best_of_many_back: loss_dict['loss_goal_back'] /= N
            loss_dict['loss_traj'] /= N    

            if self.cfg.discrepancy: loss_dict['loss_disc'] /= N

        if mode == ModeKeys.PREDICT:
            return loss_dict, pred_y, best_idxs
        else:
            return loss_dict, pred_y

    ##TODO: change the method name to `encode?`
    def forward_seq(self, spatemp_feats, input_x, input_x_frame, target_y, neighbors_x=None, cur_pos = None, first_history_indices = None, mode = None):
        N,_,_,H,W = spatemp_feats.shape  # torch.Size([n, T, C, H, W])
        # Encoded node history tensor 
        node_history_encoded = self.encode_node_history(input_x,
                                                        first_history_indices,
                                                        mode=mode)

        # Encode Relations #
        if input_x_frame is not None:
            rf_closest = torch.zeros(N,self.input_len,2).to(self.device)
            mask_rel = torch.zeros_like(spatemp_feats).to(self.device)
            for p in list(self.rf_dict.keys()): # iterate regions
                for t in range(input_x_frame.shape[1]):
                    flags = p.contains_points(input_x_frame[:,t,:])
                    if flags.sum()>0:
                        mask_rel[flags,t,:,self.rf_dict[p][0],self.rf_dict[p][1]] = 1
            # mask the untraveled regions
            rel_region = spatemp_feats * mask_rel

            # find the closest centers  
            for t in range(input_x_frame.shape[1]): # iterate time
                for i, pt in enumerate(input_x_frame[:,t,:]):
                    dist2center = torch.sum((pt - self.rf_center)**2, axis=-1)
                    if torch.isnan(dist2center).all(): # the node is unobserved at t
                        rf_closest[i, t, :] = torch.tensor([np.nan, np.nan])
                    else:
                        rf_closest[i, t, :] = (dist2center==torch.min(dist2center)).nonzero()[0]
                    # min_distance, min_poly = min(((poly.distance(point), poly) for poly in self.rf_polys), key=itemgetter(0)) 
        else:
            rel_region = spatemp_feats
 
        if self.cfg.point_rel_enable and neighbors_x is not None:
            encoded_edges_type = self.encode_edge(mode, input_x, neighbors_x, first_history_indices)
            node_rel_encoded = self.encode_total_edge_influence(mode, [encoded_edges_type,], node_history_encoded, len(node_history_encoded))
        else:
            # collect region-wise h(t) at diff. timesteps
            rel_combined = rel_region.sum((-1,-2)) # integrating the neighboring regions, torch.Size([n, T, C])
            combined_edges, _ = run_lstm_on_variable_length_seqs(self.temp_rel_enc,
                                                                original_seqs=torch.cat((rel_combined, rf_closest/10), dim=-1),
                                                                lower_indices=first_history_indices)
            # combined_edges, (_, _) = self.temp_rel_enc(torch.cat((rel_combined, rf_closest/10), dim=-1))
            combined_edges = F.dropout(combined_edges,
                               p=1. - self.cfg.rnn_dropout_keep_prob,
                               training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]
            if first_history_indices is not None:
                last_index_per_sequence = -(first_history_indices + 1)

                node_rel_encoded = combined_edges[torch.arange(first_history_indices.shape[0]), last_index_per_sequence]
            else:
                # if no first_history_indices, all sequences are full length
                node_rel_encoded = combined_edges[:, -1, :]
 
        # latent net and goal decoder (should not be dependent on relation features)
        Z, KLD = self.gaussian_latent_net(node_history_encoded, input_x[:, -1, :], target_y, z_mode=self.cfg.z_mode, mode = mode)
        enc_h_and_z = torch.cat([node_history_encoded.unsqueeze(1).repeat(1, Z.shape[1], 1), Z], dim=-1)
        pred_goal = self.goal_decoder(enc_h_and_z)

        if self.region_rel_disable:
            enc_hidden = torch.cat((node_history_encoded, node_history_encoded), dim=1) 
        else:
            enc_hidden = torch.cat((node_rel_encoded, node_history_encoded), dim=1) 
        dec_h = torch.cat([enc_hidden.unsqueeze(1).repeat(1, Z.shape[1], 1), Z], dim=-1) if self.cfg.dec_with_z else enc_hidden
        pred_traj = self.pred_future_traj(dec_h, pred_goal)
        cur_pos = input_x[:, None, -1, :self.cfg.motion.dec_output_dim] if cur_pos is None else cur_pos.unsqueeze(1)
        pred_goal = pred_goal + cur_pos
        pred_traj = pred_traj + cur_pos.unsqueeze(1)
        # compute loss
        if self.cfg.best_of_many: # no back goal estimation
            if not "".__eq__(self.cfg.latent_dist): 
                loss_goal, loss_traj = cvae_loss(pred_goal, 
                                                pred_traj, 
                                                target_y, 
                                                best_of_many=self.cfg.best_of_many) 

                loss_disc = 0 # dummy value
                if self.cfg.discrepancy and mode != ModeKeys.PREDICT:
                    loss_disc = list()
                    for enc_h_z in enc_h_and_z: # convert every agent's encoding(s) to list
                        _disc = list() 
                        # convert list of tuples to list of list
                        for _, x in enumerate(itertools.combinations(enc_h_z, r=2)): 
                            _disc.append(torch.norm(x[0]-x[1], p=2)) # x is a tuple of two encodings.
                        _disc = torch.stack(_disc)

                        loss_disc.append(_disc)
                    loss_disc = torch.stack(loss_disc) # shape: (n, C_k^2)
                    loss_disc = loss_disc.topk(k=1,largest=False)[0].log().mul(-1).mean()

                return loss_goal, loss_traj, KLD, pred_traj, loss_disc

            else:
                loss_traj = bom_traj_loss(pred_traj, target_y)
                return loss_traj, pred_traj

        elif self.cfg.z_mode:
            loss_goal, loss_traj = cvae_z_mode_loss(pred_goal, 
                                                    pred_traj, 
                                                    target_y, 
                                                    z_mode=self.cfg.z_mode) 

            return loss_goal, loss_traj, KLD, pred_traj

        elif self.cfg.best_of_many_back:
            enc_h_back = []  #  cur_pos: torch.Size([1, 1, 2])
            for k in range(pred_traj.shape[2]):
                outputs, _ = run_lstm_on_variable_length_seqs(
                    self.traj_enc_back,
                    original_seqs=torch.flip(pred_traj[:,:,k,:], dims=(1,)))

                outputs = F.dropout(outputs,
                                    p=1. - self.cfg.rnn_dropout_keep_prob,
                                    training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]

                enc_h_back.append(outputs[:, -1, :])

            enc_h_back = torch.stack(enc_h_back, dim=1)
            pred_goal_back = self.goal_decoder_back(enc_h_back) 
            pred_goal_back = pred_goal_back + cur_pos

            loss_goal, loss_goal_back, loss_traj, best_idx = cvae_back_loss(pred_goal, 
                                                                 pred_goal_back,
                                                                 pred_traj, 
                                                                 input_x[torch.arange(first_history_indices.shape[0]), first_history_indices],
                                                                 target_y, 
                                                                 best_of_many_back=self.cfg.best_of_many_back) 

            return loss_goal, loss_goal_back, loss_traj, KLD, pred_traj, best_idx

    def gaussian_latent_net(self, enc_h, cur_state, target=None, z_mode=None, mode = None):
        """
        :param enc_h: Input / Condition tensor.
        :param cur_state: Input tensor including the state for each agent at current time [bs, state].
        :param target: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :return: tuple(z, kl_obj)
            WHERE
            - z: Samples from the latent space.
            - kl_obj: KL Divergenze between q and p
        """
        # get mu, sigma
        # 1. sample z from piror
        z_mu_logvar_p = self.p_z_x(enc_h)
        z_mu_p = z_mu_logvar_p[:, :self.cfg.latent_dim]
        z_logvar_p = z_mu_logvar_p[:, self.cfg.latent_dim:]
        if target is not None and mode != ModeKeys.PREDICT:
            # 2. sample z from posterior, for training only
            initial_h = self.node_future_encoder_h(cur_state)
            initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=initial_h.device)], dim=0)
            _, target_h = self.gt_goal_encoder(target, initial_h)
            target_h = target_h.permute(1,0,2)
            target_h = target_h.reshape(-1, target_h.shape[1] * target_h.shape[2])
            
            target_h = F.dropout(target_h,
                                p=1.-self.cfg.rnn_dropout_keep_prob,
                                training=self.training)

            z_mu_logvar_q = self.q_z_xy(torch.cat([enc_h, target_h], dim=-1))
            z_mu_q = z_mu_logvar_q[:, :self.cfg.latent_dim]
            z_logvar_q = z_mu_logvar_q[:, self.cfg.latent_dim:]
            Z_mu = z_mu_q
            Z_logvar = z_logvar_q

            # 3. compute KL(q_z_xy||p_z_x)
            KLD = 0.5 * ((z_logvar_q.exp()/z_logvar_p.exp()) + \
                        (z_mu_p - z_mu_q).pow(2)/z_logvar_p.exp() - \
                        1 + \
                        (z_logvar_p - z_logvar_q))
            KLD = KLD.sum(dim=-1).mean()
            KLD = torch.clamp(KLD, min=0.001)
        else:
            Z_mu = z_mu_p
            Z_logvar = z_logvar_p
            KLD = 0.0
        
        # 4. Draw sample
        K_samples = torch.randn(enc_h.shape[0], self.cfg.k, self.cfg.latent_dim).to(self.device)
        Z_std = torch.exp(0.5 * Z_logvar)
        Z = Z_mu.unsqueeze(1).repeat(1, self.cfg.k, 1) + K_samples * Z_std.unsqueeze(1).repeat(1, self.cfg.k, 1)

        if z_mode:
            Z = torch.cat((Z_mu.unsqueeze(1), Z), dim=1)
        return Z, KLD 
    
    def pred_future_traj(self, dec_h, G):
        '''
        use a bidirectional GRU decoder to plan the path.
        Params:
            dec_h: (Batch, hidden_dim) if not using Z in decoding, otherwise (Batch, K, dim) 
            G: (Batch, K, pred_dim)
        Returns:
            backward_outputs: (Batch, T, K, pred_dim)
        '''
        pred_len = self.cfg.pred_len

        K = G.shape[1]
        # 1. run forward
        forward_outputs = []
        forward_h = self.enc_h_to_forward_h(dec_h)
        if len(forward_h.shape) == 2:
            forward_h = forward_h.unsqueeze(1).repeat(1, K, 1)
        forward_h = forward_h.view(-1, forward_h.shape[-1])
        forward_input = self.traj_dec_input_forward(forward_h)
        for t in range(pred_len): # the last step is the goal, no need to predict
            forward_h = self.traj_dec_forward(forward_input, forward_h)
            forward_input = self.traj_dec_input_forward(forward_h)
            forward_outputs.append(forward_h)
        
        forward_outputs = torch.stack(forward_outputs, dim=1)

        if self.cfg.dec_backward_disable:
            forward_outputs = self.traj_output(forward_outputs)
            forward_outputs = forward_outputs.view(-1, K, pred_len, forward_outputs.shape[-1])

            return torch.transpose(forward_outputs, 1, 2)
        # 2. run backward on all samples
        backward_outputs = []
        backward_h = self.enc_h_to_back_h(torch.cat((dec_h[:,self.cfg.motion.enc_hidden_size:], dec_h[:,self.cfg.motion.enc_hidden_size:]), dim=1)) 
        if len(dec_h.shape) == 2:
            backward_h = backward_h.unsqueeze(1).repeat(1, K, 1)
        backward_h = backward_h.view(-1, backward_h.shape[-1])
        backward_input = self.traj_dec_input_backward(G)#torch.cat([G])
        backward_input = backward_input.view(-1, backward_input.shape[-1])
        
        for t in range(pred_len-1, -1, -1):
            backward_h = self.traj_dec_backward(backward_input, backward_h)
            output = self.traj_output(torch.cat([backward_h, forward_outputs[:, t]], dim=-1))
            backward_input = self.traj_dec_input_backward(output)
            backward_outputs.append(output.view(-1, K, output.shape[-1]))
        
        # inverse because this is backward 
        backward_outputs = backward_outputs[::-1]
        backward_outputs = torch.stack(backward_outputs, dim=1)
        
        return backward_outputs

    def encode_node_history(self, node_hist, first_history_indices=None, mode=None):
        """
        Encodes the nodes history.
        adopted from https://github.com/umautobots/bidireaction-trajectory-prediction/blob/main/bitrap/modeling/bitrap_np.py#L177
        :param node_hist: Historic and current state of the node. [bs, mhl, state]
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :return: Encoded node history tensor. [bs, enc_rnn_dim]
        """
        # outputs, _ = run_lstm_on_variable_length_seqs(self.node_modules[self.node_type + '/node_history_encoder'],
        #                                               original_seqs=node_hist,
        #                                               lower_indices=first_history_indices)
        outputs, _ = self.encode_variable_length_seqs(node_hist,
                                                      lower_indices=first_history_indices)
        outputs = F.dropout(outputs,
                            p=1. - self.cfg.rnn_dropout_keep_prob,
                            training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]

        if first_history_indices is not None:
            last_index_per_sequence = -(first_history_indices + 1)

            return outputs[torch.arange(first_history_indices.shape[0]), last_index_per_sequence]
        else:
            # if no first_history_indices, all sequences are full length
            return outputs[:, -1, :]

    def encode_variable_length_seqs(self, original_seqs, lower_indices=None, upper_indices=None, total_length=None):
        '''
        take the input_x, pack it to remove NaN, embed, and run GRU
        adopted from https://github.com/umautobots/bidireaction-trajectory-prediction/blob/main/bitrap/modeling/bitrap_np.py#L144
        '''
        bs, tf = original_seqs.shape[:2]
        if lower_indices is None:
            lower_indices = torch.zeros(bs, dtype=torch.int)
        if upper_indices is None:
            upper_indices = torch.ones(bs, dtype=torch.int) * (tf - 1)
        if total_length is None:
            total_length = max(upper_indices) + 1
        # This is done so that we can just pass in self.prediction_timesteps
        # (which we want to INCLUDE, so this will exclude the next timestep).
        inclusive_break_indices = upper_indices + 1
        pad_list = []
        length_per_batch = []
        for i, seq_len in enumerate(inclusive_break_indices):
            pad_list.append(original_seqs[i, lower_indices[i]:seq_len])
            length_per_batch.append(seq_len-lower_indices[i])
 
        # 1. embed and convert back to pad_list
        x = self.hist_enc_embed(torch.cat(pad_list, dim=0))
        pad_list = torch.split(x, length_per_batch)

        # 2. run temporal
        packed_seqs = rnn.pack_sequence(pad_list, enforce_sorted=False) 
        packed_output, h_x = self.hist_encoder(packed_seqs)
        # pad zeros to the end so that the last non zero value 
        output, _ = rnn.pad_packed_sequence(packed_output,
                                            batch_first=True,
                                            total_length=total_length)
        return output, h_x

    def encode_edge(self,
                    mode,
                    node_history_st,
                    neighbors,
                    # neighbors_edge_value,
                    first_history_indices):
        assert len(node_history_st)==len(neighbors)
        max_hl = self.cfg.maximum_history_length

        edge_states_list = list()  # list of [#of neighbors, max_ht, state_dim]
        for i, neighbor_states_dict in enumerate(neighbors):
            neighbor_states = list(neighbor_states_dict.values())[0]  #list of neighbors for the current agent
            if len(neighbor_states) == 0:  # There are no neighbors for edge type # TODO necessary?
                neighbor_state_length = int(
                    np.sum([len(entity_dims) for entity_dims in self.state['Pedestrian'].values()])
                )
                edge_states_list.append(torch.zeros((1, max_hl + 1, neighbor_state_length), device=self.device))
            else:
                edge_states_list.append(torch.stack(neighbor_states, dim=0).to(self.device))

        if self.cfg.edge_state_combine_method == 'sum':
            # Used in Structural-RNN to combine edges as well.
            op_applied_edge_states_list = list()
            for neighbors_state in edge_states_list:
                op_applied_edge_states_list.append(torch.sum(neighbors_state, dim=0))
            combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0)
            # if self.hyperparams['dynamic_edges'] == 'yes':
            #     # Should now be (bs, time, 1)
            #     op_applied_edge_mask_list = list()
            #     for edge_value in neighbors_edge_value:
            #         op_applied_edge_mask_list.append(torch.clamp(torch.sum(edge_value.to(self.device),
            #                                                                dim=0, keepdim=True), max=1.))
            #     combined_edge_masks = torch.stack(op_applied_edge_mask_list, dim=0)

        joint_history = torch.cat([combined_neighbors, node_history_st], dim=-1)

        outputs, _ = run_lstm_on_variable_length_seqs(
            self.edge_encoder,
            original_seqs=joint_history,
            lower_indices=first_history_indices
        )

        outputs = F.dropout(outputs,
                            p=1. - self.cfg.rnn_dropout_keep_prob,
                            training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]

        last_index_per_sequence = -(first_history_indices + 1)
        ret = outputs[torch.arange(last_index_per_sequence.shape[0]), last_index_per_sequence]
        # if self.hyperparams['dynamic_edges'] == 'yes':
        #     return ret * combined_edge_masks
        # else:
        #     return ret
        return ret

    def encode_total_edge_influence(self, mode, encoded_edges, node_history_encoder, batch_size):
        # Used in Social Attention (https://arxiv.org/abs/1710.04689)
        if len(encoded_edges) == 0:
            combined_edges = torch.zeros((batch_size, self.edge_encoder.hidden_size), device=self.device)

        else:
            # axis=1 because then we get size [batch_size, max_time, dim]
            encoded_edges = torch.stack(encoded_edges, dim=1)
            combined_edges, _ = self.edge_influence_encoder(encoded_edges, node_history_encoder)
            combined_edges = F.dropout(combined_edges,
                                       p=1. - self.cfg.rnn_dropout_keep_prob,
                                       training=(mode == ModeKeys.TRAIN))

        return combined_edges
