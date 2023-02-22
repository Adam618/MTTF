import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class ImgEmbedding(nn.Module):
    def __init__(self, c_in, img_channel,d_model):
        super(ImgEmbedding, self).__init__()

        self.ImgConv = nn.Conv3d(in_channels=img_channel, out_channels=128,
                                 kernel_size=(3, 16, 16), padding=(1, 0, 0),stride=(1,16,16))
        self.ImgLinear = torch.nn.Linear(3200, d_model)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.permute(0,2,1,3,4)
        x = self.ImgConv(x)
#         x = np.squeeze(x, 1)
        x = x.permute(0,2,1,3,4)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # x = x.view(-1)
        x = self.ImgLinear(x)
        return x




    
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self,name,is_img_embed, img_channel, c_in, d_model, cross_attention,embed_type='fixed', freq='h', dropout=0.1):
#         d_model_1 = d_model
#         if is_img_embed and name == 'decoder':
#             d_model = int(d_model/2)
#             d_model_1 = d_model*2
        super(DataEmbedding, self).__init__()
        self.name = name
        self.is_img_embed = is_img_embed
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.Img_embedding = ImgEmbedding(c_in=c_in,img_channel= img_channel,d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

        self.feature_fusion = nn.Sequential(
        nn.BatchNorm1d(14),
        nn.Linear(d_model*2, d_model, bias=False),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(d_model,d_model*2)
    )
        self.cross_attention = cross_attention


    def forward(self, img_data, x, x_mark):
        
         
        if self.name == 'series':
            x = self.value_embedding(x) + self.position_embedding(x) 
        elif self.name == 'img':
            x = self.Img_embedding(img_data) + self.position_embedding(img_data)
        else:
            x = self.value_embedding(x) + self.position_embedding(x) 
        return self.dropout(x)