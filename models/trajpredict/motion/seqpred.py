import torch.nn as nn
import torch
import copy
import numpy as np
import pdb
 

class constVelocity(nn.Module):
    def __init__(self, cfg): 
        self.obs_length=cfg.INPUT_LEN
        self.n_predict = cfg.pred_len
    
    def forward(self, x):
        n_predict = self.n_predict
        curr_position = x[-1]  # dim
        curr_velocity = x[-1] - x[-2]
        output_rel_scenes = np.array([i * curr_velocity for i in range(1, n_predict+1)])  # displacement
        output_scenes = curr_position + output_rel_scenes
        return output_scenes


class LSTM(nn.Module):
    def __init__(self, cfg):
        super(LSTM, self).__init__()
        self.pred_len = cfg.pred_len
        self.cfg = copy.deepcopy(cfg.motion)
        self.enc_embed = nn.Sequential(nn.Linear(self.cfg.global_input_dim, self.cfg.enc_input_size), 
                                        nn.ReLU())
        ## LSTMs
        self.encoder = torch.nn.LSTM(self.cfg.enc_input_size, self.cfg.enc_hidden_size, batch_first=True)
        self.decoder = torch.nn.LSTMCell(self.cfg.dec_input_size, self.cfg.enc_hidden_size)
        self.dec_embed = nn.Linear(self.cfg.global_input_dim, self.cfg.dec_input_size) 
        self.out = nn.Linear(self.cfg.enc_hidden_size, self.cfg.dec_output_dim)
    
    def forward(self, x):
        N = x.shape[0]
        traj_feats = self.enc_embed(x)   
        _, (hidden, cx) = self.encoder(traj_feats)
        outputs = torch.zeros(N, self.pred_len, self.cfg.dec_output_dim).to('cuda')
        dec_input = x[:,-1,:] ## last position 
        dec_input = self.dec_embed(dec_input) # N,dim
        
        hidden, cx = hidden.squeeze(0), cx.squeeze(0)   
        for t in range(self.pred_len):
            hidden, cx = self.decoder(dec_input, (hidden, cx))  ## the last hidden of encoder as initial hidden
            out = self.out(hidden)  # N, 4
            outputs[:,t,:] = out
            dec_input = self.dec_embed(out)
        return outputs

class GRU(nn.Module):
    """
    single GRU that takes bbox coordinates as inputs and predict future bbox coordinates (no interaction modeling)
    """
    def __init__(self, cfg):
        super(GRU, self).__init__()
        self.pred_len = cfg.pred_len
        self.cfg = copy.deepcopy(cfg.motion)
        ## raw (normalized) input trajectory --> trajectory embedding
        self.enc_embed = nn.Sequential(nn.Linear(self.cfg.global_input_dim, self.cfg.enc_input_size), 
                                        nn.ReLU())
        self.encoder = EncoderRNN(self.cfg.enc_input_size, self.cfg.enc_hidden_size, 1)
#         self.dec_embed = nn.Linear(self.cfg.global_input_dim, self.cfg.dec_input_size) # raw input trajectory --> trajectory embedding
        self.dec_embed = nn.Sequential(nn.Linear(self.cfg.enc_hidden_size, self.cfg.dec_input_size), 
                                        nn.ReLU())
        self.decoder = DecoderRNN(self.cfg.dec_input_size, self.cfg.dec_hidden_size, self.cfg.dec_output_dim, 1)
        
    def forward(self, x):
        N = x.shape[0]
        traj_feats = self.enc_embed(x)   
        enc_hidden, _ = self.encoder(traj_feats)  # (batch_size, obs_length, enc_hidden)
        outputs = torch.zeros(N, self.pred_len, self.cfg.dec_output_dim).to('cuda')
#         dec_input = x[:,-1,:].unsqueeze(1) ## last position 
        hidden = enc_hidden[:,-1,:] # last hidden
        dec_input = self.dec_embed(hidden).unsqueeze(1) 
        hidden = hidden.unsqueeze(0)
        for t in range(self.pred_len):
            out, hidden = self.decoder(dec_input, hidden)  ## the last hidden of encoder as initial hidden
            outputs[:,t,:] = out.squeeze(1) 
            dec_input = self.dec_embed(hidden.squeeze(0)).unsqueeze(1)
        return outputs
    
    
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, x):
        out, hidden = self.rnn(x)
        return out, hidden
    
class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_embed, hidden):
        dec_output, dec_hidden = self.rnn(input_embed, hidden)
        dec_output = self.out(dec_output)
        return dec_output, dec_hidden    
    
    
if __name__=='__main__':
    import pickle
    with open(f'cfg.pkl' , 'rb') as f:
        cfg= pickle.load(f)
    gru = GRU(cfg.model)
#     lstm = LSTM(cfg.model)
#     constvel = constVelocity(cfg.model)
    samples = torch.ones([10, cfg.model.input_len, cfg.model.global_input_dim])
#     for x in samples:
#         print(constvel.predict(x).shape)

    print(gru(samples).shape)