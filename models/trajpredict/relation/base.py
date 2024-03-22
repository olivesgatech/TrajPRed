import os
import sys
# sys.path.append(os.path.realpath('.'))
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
from configs import cfg
from models.dimreduct.convae import CAE
from models.trajpredict.motion.seqpred import EncoderRNN, DecoderRNN 


class TemporalEncoder(nn.Module):
    def __init__(self, cfg):
        super(TemporalEncoder, self).__init__()
        self.conv1 = nn.Conv3d(cfg.convae.latent_dim, cfg.convae.latent_dim, kernel_size=(cfg.input_len//2, 1, 1), stride=(2, 1, 1), padding=0)
        self.conv2 = nn.Conv3d(cfg.convae.latent_dim, cfg.convae.latent_dim, kernel_size=(3, 1, 1), padding=0)
        # self.batchNorm = nn.BatchNorm2d(cfg.convae.latent_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class TemporalDecoder(nn.Module):
    def __init__(self, cfg):
        super(TemporalDecoder, self).__init__()
        self.deconv1 = nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=(3, 1, 1), padding=0,)
        self.deconv2 = nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0,)
        self.deconv3 = nn.ConvTranspose3d(in_channels=128, out_channels=cfg.convae.latent_dim, kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=0,)
        
    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x

class BaseModel(nn.Module):
    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        # construct a density reconstruction model (convAE)
        self.cae = CAE(in_channel=1)
        # global configs
        self.input_len = cfg.input_len
        self.pred_len = cfg.pred_len
        # # (global) intent configs
        # self.pred_intent = cfg.pred_intent
        # if cfg.pred_intent:
        #     self.npc = cfg.num_pedintent_class
        # motion configs
        self.cfg_mot = copy.deepcopy(cfg.motion)
        if cfg.tempenc == 'rnn':
            if cfg.region_rel_backbone == 'GRU':
                self.temp_enc = nn.GRU(input_size=cfg.convae.latent_dim, hidden_size=cfg.motion.enc_hidden_size, batch_first=True)
            else:
                self.temp_enc = nn.LSTM(input_size=cfg.convae.latent_dim, hidden_size=cfg.motion.enc_hidden_size, batch_first=True)
        elif cfg.tempenc == 'conv':        
            self.temp_enc = TemporalEncoder(cfg)
        if cfg.tempdec == 'rnn': 
            self.temp_dec = nn.GRUCell(input_size=128+cfg.convae.latent_dim, hidden_size=cfg.tempdec_hidden_size)
            self.temp_dec_out = nn.Sequential(nn.Linear(cfg.tempdec_hidden_size, cfg.convae.latent_dim), nn.ReLU())
        elif cfg.tempdec == 'deconv': 
            self.temp_dec = TemporalDecoder(cfg)
        # history encoder 
        self.hist_enc_embed = nn.Sequential(nn.Linear(self.cfg_mot.global_input_dim, self.cfg_mot.enc_input_size), nn.ReLU())
        self.hist_encoder = EncoderRNN(self.cfg_mot.enc_input_size, self.cfg_mot.enc_hidden_size, 1) 
        # future decoder
        # self.fut_dec_embed = nn.Sequential(nn.Linear(self.cfg_mot.enc_hidden_size, self.cfg_mot.dec_input_size), nn.ReLU())
        # self.fut_decoder = DecoderRNN(self.cfg_mot.dec_input_size, self.cfg_mot.dec_hidden_size, self.cfg_mot.dec_output_dim, 1)
        self.fut_dec_embed = nn.Sequential(nn.Linear(self.cfg_mot.dec_output_dim, self.cfg_mot.dec_input_size), nn.ReLU(),)
        self.fut_decoder = nn.GRUCell(input_size=self.cfg_mot.dec_input_size+self.cfg_mot.enc_hidden_size+128, hidden_size=self.cfg_mot.dec_hidden_size)
        self.fut_out = nn.Linear(self.cfg_mot.dec_hidden_size, self.cfg_mot.dec_output_dim)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
 