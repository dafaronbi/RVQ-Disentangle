# Ported from: https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb.

import torch

from torch import nn
from torch.nn import functional as F
from torch.nn.init import kaiming_uniform, normal
import argparse
import logging
import os
import random

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torch.utils.tensorboard
from torch import nn
from custom_layers import nn_custom, vq_custom
import dac
import math
import einops
from einops.layers.torch import Rearrange

class Mean(nn.Module):
    def forward(self,x):
        return torch.mean(x, -2)

class FilterGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, filter_generator_channels):
        super(FilterGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.filter_generator = nn.Sequential(
            nn.Conv1d(in_channels, filter_generator_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(filter_generator_channels, out_channels * in_channels * kernel_size , kernel_size=1)
        )

    def forward(self, x):
        batch_size, _, length = x.size()
        # Generate dynamic filters
        filters = self.filter_generator(x)
        filters = filters.view(batch_size, self.out_channels, self.in_channels, self.kernel_size, length)
        filters = filters.mean(0)
        filters = filters.mean(-1)

        #Shape filter
        conv_weights = filters
        trans_weights = einops.rearrange(filters, "o i k -> i o k")

        return conv_weights,trans_weights

class ResBlock(nn.Module):
    def __init__(self, filter_size):
        super().__init__()

        self.conv1 = nn.Conv1d(filter_size, filter_size, 1, 1, 0)
        self.conv2 = nn.Conv1d(filter_size, filter_size, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm1d(filter_size, affine=False)

        self.reset()

    def forward(self, input, gamma, beta):
        out = self.conv1(input)
        resid = F.relu(out)
        out = self.conv2(resid)
        out = self.bn(out)

        gamma = gamma.unsqueeze(2)
        beta = beta.unsqueeze(2)

        out = gamma * out + beta

        out = F.relu(out)
        out = out + resid

        return out

    def reset(self):
        kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.zero_()
        kaiming_uniform(self.conv2.weight)

class bNorm(nn.Module):
    def __init__(self, num_features,device):
        super().__init__()

        if device == torch.device('cpu'):
            self.bnLayer = nn.BatchNorm1d(num_features)
        else:
            self.bnLayer = nn.SyncBatchNorm(num_features)
    
    def forward(self, inp):
        return self.bnLayer(inp)

class attention_block(nn.Module):
    def __init__(self, embed_dim, num_heads, device):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.device = device
    
    def forward(self, inp):
        
        output = torch.zeros((inp.shape[0], 0, inp.shape[2], inp.shape[3])).to(self.device)

        for i in range(inp.shape[1]):
            out,_ = self.multihead_attn(inp[:,i,...],inp[:,i,...],inp[:,i,...])
            out = out.unsqueeze(1)
            output = torch.cat((output, out), dim=1)
        return output

class transformer_block(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, device):
        super().__init__()

        self.proj = nn.Sequential(
                nn.Linear(input_size,d_model),
                nn.ReLU(),
        )

        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True), num_layers=num_layers)
        self.device = device
        
    
    def forward(self, inp):
        
        #make positional embeddings
        max_len = inp.shape[2]
        d_model = inp.shape[1]
        batch_size = inp.shape[0]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
        sz = inp.shape[-1]
        # mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        inp = inp + pe
        inp = einops.rearrange(inp, "b d t -> b t d")
        output = self.proj(inp)
        output = self.transformer(output)
        
        return output

class cross_attention_block(nn.Module):
    def __init__(self, embed_dim, output_size, num_heads, device):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.device = device
        self.proj = nn.Sequential(
                nn.ReLU(),
                nn.Linear(embed_dim, output_size),
                Rearrange("b t f (d c) -> b d c (t f)", d=1024),
                nn.LogSoftmax(dim=1),
        )
    
    def forward(self, inp, features):

        inp = einops.rearrange(inp, "b d (t f)-> b t f d", f=8)
        features = einops.rearrange(features, "b d (t f)-> b t f d", f=8)
        
        output = torch.zeros((inp.shape[0], 0, inp.shape[2], inp.shape[3])).to(self.device)

        for i in range(inp.shape[1]):
            out,_ = self.multihead_attn(inp[:,i,...],features[:,i,...], features[:,i,...])
            out = out.unsqueeze(1)
            output = torch.cat((output, out), dim=1)

        output = self.proj(output)

        return output

class transformer_decoder_block(nn.Module):
    def __init__(self, d_model, output_size, num_heads, num_layers, device):
        super().__init__()

        self.transformer = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, batch_first=True), num_layers=num_layers)
        self.device = device
        self.proj = nn.Sequential(
                nn.Linear(d_model, output_size),
                Rearrange("b t (d c) -> b d c t", d=1024),
                nn.LogSoftmax(dim=1),
        )
        
    
    def forward(self, inp, features):
        #make positional embeddings
        max_len = inp.shape[2]
        d_model = inp.shape[1]
        batch_size = inp.shape[0]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
        sz = inp.shape[-1]
        # mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        inp = inp + pe
        inp = einops.rearrange(inp, "b d t -> b t d")

        max_len = features.shape[2]
        d_model = features.shape[1]
        batch_size = inp.shape[0]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)

        features = features + pe
        features = einops.rearrange(features, "b d t -> b t d")

        output = self.transformer(inp,features)
        output = self.proj(output)
        
        return output

        
class new_model(nn.Module):
    def __init__(self, device, parameters):
        super().__init__()

        self.device = device
        self.training_params = parameters
        self.emb_dim = self.training_params["emb_dim"]

        #pitch countour (pc) numerator and denominator
        self.pc_num = 3
        self.pc_denom = 4

        self.pitch_emb = None
        self.rest_emb = None

        self.temperature = 1

        self.num_heads = self.training_params["num_heads"]


        if self.training_params["codebook"] == "learned":
            self.input_emb_dim = 512
        
        if self.training_params["codebook"] == "default":
            self.input_emb_dim = 1024

        if self.training_params["input_type"] == "continuous":
            self.input_emb_dim = 1024
        
        if self.training_params["disentangle"] == "pesto":
            self.input_emb_dim = 657

        if self.training_params["input_dim"]:
            self.input_emb_dim = self.training_params["input_dim"]

        if self.training_params["disentangle"] == "pesto":
            self.pesto_emb = torch.load("note_to_emb.pt")
            self.emb_dim = 657
            self.num_heads = 73

        if self.training_params["encoder_type"] == "stride":
            self.enc = nn.Sequential(
            nn.Conv1d(in_channels=self.input_emb_dim, out_channels=self.emb_dim, kernel_size=16, stride=16, padding = 8, dilation=1),
            bNorm(self.emb_dim,self.device),
            nn.ReLU(),
            )

        if self.training_params["encoder_type"] == "no_stride":
            self.enc = nn.Sequential(
            nn.Conv1d(in_channels=self.input_emb_dim, out_channels=self.emb_dim, kernel_size=5, stride=1, padding =2, dilation=1),
            bNorm(self.emb_dim,self.device),
            nn.ReLU(),
            )

        if self.training_params["encoder_type"] == None:
            self.enc = nn.Identity()

        if self.training_params["encoder_type"] == "mlp":
            self.enc = nn.Sequential(
                nn_custom.SwapAxes((1,2)),
                nn.Linear(self.input_emb_dim,self.emb_dim),
                nn.ReLU(),
                nn_custom.SwapAxes((1,2)),
            )

        if self.training_params["encoder_type"] == "gru_only":
            self.enc = nn.Sequential(
                nn_custom.GRUWrap(self.input_emb_dim,self.emb_dim,2, batch_first=True),
                nn.ReLU(),
            )
        
        if self.training_params["encoder_type"] == "attention":
            self.enc = nn.Sequential(
                Rearrange("b d (t f)-> b t f d", f=8),
                attention_block(self.emb_dim, self.num_heads, self.device),
                nn.ReLU(),
                Rearrange("b t f d -> b d (t f)"),
                )
        
        if self.training_params["encoder_type"] == "transformer":
            self.enc = nn.Sequential(
                Rearrange("b d t -> b t d"),
                nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=self.num_heads, batch_first=True), num_layers=self.training_params["num_enc_layers"]),
                Rearrange("b t d -> b d t"),
            )

        self.p_enc = torch.nn.Embedding(128,self.emb_dim)

        self.dec = nn.Sequential()

        if self.training_params["codes"] == "flattened":
            token_size = 1024
        else:
            token_size = 1024*9

        if self.training_params["input_type"] == "continuous":
            token_size = 1024

        d_size = self.emb_dim
        if self.training_params["disentangle"] == "concat" or self.training_params["disentangle"] == "pesto":
            d_size = self.emb_dim*2
        if self.training_params["disentangle"] == "across_inst" or self.training_params["disentangle"] == "random_inst":
            d_size = self.emb_dim*3
        
        if self.training_params["encoder_type"] == "stride":
            self.dec = nn.Sequential(
                nn.ConvTranspose1d(in_channels=d_size, out_channels=d_size, kernel_size=16, stride=16, padding=4, dilation=1),
                bNorm(d_size,self.device),
                nn.ReLU()
            )
        
            
        # if self.training_params["disentangle"] == "concat" or self.training_params["disentangle"] == "pesto":
        #     d_size = self.emb_dim*2
        #     self.dec = nn.Sequential(
        #         nn.ConvTranspose1d(in_channels=self.emb_dim*2, out_channels=self.emb_dim*2, kernel_size=16, stride=16, padding=4, dilation=1),
        #         bNorm(self.emb_dim*2,self.device),
        #         nn.ReLU())
        # elif self.training_params["disentangle"] == "across_inst" or self.training_params["disentangle"] == "random_inst":
        #     d_size = self.emb_dim*3
        #     self.dec = nn.Sequential(
        #         nn.ConvTranspose1d(in_channels=self.emb_dim*3, out_channels=self.emb_dim*3, kernel_size=16, stride=16, padding=4, dilation=1),
        #         bNorm(self.emb_dim*3,self.device),
        #         nn.ReLU()
        #     )

        dilation_index = [1,3,6,9,12,15]
        for i in range(self.training_params["num_dec_layers"]):
            if i == 0:
                first_dim = d_size
            else:
                first_dim = 256
            self.dec = self.dec + nn.Sequential(
                nn.Conv1d(in_channels=first_dim, out_channels=256, kernel_size=7, stride=1, padding=3*dilation_index[i], dilation=dilation_index[i]),
                bNorm(256,self.device),
                nn.ReLU())

        if self.training_params["decoder_type"] == "no_gru":
            self.dec = self.dec + nn.Sequential(
                # nn_custom.GRUWrap(256,256,1, batch_first=True),
                nn.ReLU(),
                nn_custom.SwapAxes((1,2)),
                nn.Linear(256,512),
                nn.ReLU(),
                nn.Linear(512,1024),
                nn.ReLU(),
                nn.Linear(1024,token_size),
                nn.ReLU(),
                nn_custom.SwapAxes((1,2)),
                nn.Unflatten(1, (1024,token_size//1024)),
                nn.LogSoftmax(dim=1),
                )

        if self.training_params["decoder_type"] == "gru":
            self.dec = self.dec + nn.Sequential(
                nn_custom.GRUWrap(256,256,1, batch_first=True),
                nn.ReLU(),
                nn_custom.SwapAxes((1,2)),
                nn.Linear(256,512),
                nn.ReLU(),
                nn.Linear(512,1024),
                nn.ReLU(),
                nn.Linear(1024,token_size),
                nn.ReLU(),
                nn_custom.SwapAxes((1,2)),
                nn.Unflatten(1, (1024,token_size//1024)),
                nn.LogSoftmax(dim=1),
                )

        if self.training_params["decoder_type"] == "mlp":
            self.dec = self.dec + nn.Sequential(
                nn_custom.SwapAxes((1,2)),
                nn.Linear(self.emb_dim,1024),
                nn.ReLU(),
                nn_custom.SwapAxes((1,2)),
                )
        
        if self.training_params["decoder_type"] == "mlp_discrete":
            self.dec = self.dec + nn.Sequential(
                nn_custom.SwapAxes((1,2)),
                nn.Linear(self.emb_dim,1024*9),
                nn.ReLU(),
                Rearrange("b t (d c) -> b d c t", d=1024),
                nn.LogSoftmax(dim=1),
                )
        
        if self.training_params["decoder_type"] == "attention":
            self.dec = self.dec + nn.Sequential(
                Rearrange("b d (t f)-> b t f d", f=8),
                attention_block(self.emb_dim, self.num_heads, self.device),
                nn.Linear(self.emb_dim,1024*9),
                nn.ReLU(),
                Rearrange("b t f (d c)-> b d c (t f)", d=1024),
                nn.LogSoftmax(dim=1),
                )
        
        if self.training_params["decoder_type"] == "gru_only":
            self.dec = nn.Sequential(
                nn_custom.GRUWrap(self.input_emb_dim,self.input_emb_dim,2, batch_first=True),
                nn.ReLU(),
                Rearrange("b d t -> b t d "),
                nn.Linear(self.input_emb_dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024*9),
                nn.ReLU(),
                # nn_custom.SwapAxes((1,2)),
                # nn.Unflatten(1, (1024,9)),
                Rearrange("b t (d c) -> b d c t", d=1024),
                nn.LogSoftmax(dim=1),
                # Rearrange("b t d c -> b d c t"),
            )
        
        if self.training_params["decoder_type"] == "transformer":
            self.dec = nn.Sequential(
                # nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=64, batch_first=True), num_layers=self.training_params["num_dec_layers"]),
                transformer_block(d_size, self.emb_dim, self.num_heads, self.training_params["num_dec_layers"], self.device),
                torch.nn.Linear(self.emb_dim,1024*9),
                Rearrange("b t (d c) -> b d c t", d=1024),
                nn.LogSoftmax(dim=1),
            )
        
        if self.training_params["decoder_type"] == "transformer_decoder":
            self.dec = transformer_decoder_block(d_size, 1024*9, self.num_heads, self.training_params["num_dec_layers"], self.device)
        
        if self.training_params["decoder_type"] == "cross_attention":
            self.dec = cross_attention_block(d_size, 1024*9, self.num_heads, self.device)


        self.proj_matrix = [torch.nn.Embedding(1025,self.input_emb_dim).to(self.device) for i in range(9)]

        if  self.training_params["disentangle"] == "conv" or self.training_params["disentangle"] == "conv_single":
            k_size = self.training_params["d_conv_size"]
            self.rest_conv = nn.Conv1d(in_channels=self.emb_dim*2, out_channels=self.emb_dim, kernel_size=k_size, padding=k_size//2,  stride=1)
            self.transform_conv = nn.Conv1d(in_channels=self.emb_dim*2, out_channels=self.emb_dim, kernel_size=k_size, padding=k_size//2, stride=1)

        if  self.training_params["disentangle"] == "FiLM":
            self.resblocks = nn.ModuleList()
            self.n_FiLM = self.training_params["n_FiLM"]
            for i in range(self.n_FiLM):
                self.resblocks.append(ResBlock(self.emb_dim))
                self.film = nn.Linear(self.emb_dim, self.emb_dim * 2 * self.n_FiLM)

        if self.training_params["disentangle"] == "across_inst" or self.training_params["disentangle"] == "random_inst":
            self.t_enc = nn.Sequential(
            nn.Conv1d(in_channels=self.input_emb_dim, out_channels=self.emb_dim, kernel_size=16, stride=16, dilation=1),
            bNorm(self.emb_dim,self.device),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.emb_dim, out_channels=self.emb_dim, kernel_size=21, dilation=1),
            bNorm(self.emb_dim,self.device),
            nn.ReLU(),
            )

        self.filter_generator = FilterGenerator(32, 32, 1, 32)
        
    
    def forward(self, dacModel, z,p, z_prime,p_prime):

        #loss categorical and regression loss functions
        if self.training_params["loss_type"] == "hierarchical_nll":
            C_loss = nn.NLLLoss(reduction='none')
        else:
            C_loss = nn.NLLLoss()

        COS_loss = nn.CosineEmbeddingLoss()
        MSE_loss = nn.MSELoss()
        
        #convert codes to embedding dimensions
        z_codes = z
        with torch.no_grad():
            z = dacModel.quantizer.from_codes(z)[0]

        z_prime_codes = z_prime
        z_prime_codes_saved = z_prime_codes
        with torch.no_grad():
            z_prime, z_prime_latents,_ = dacModel.quantizer.from_codes(z_prime)
        
        # for i in range(9):
        #     print(dacModel.quantizer.quantizers[i].codebook.weight.shape)
        # exit()

        #get latent from audio input
        d_emb = torch.zeros(z.shape[0],  z.shape[-1], 0,self.input_emb_dim).to(self.device)

        for i in range(9):
            if self.training_params["codebook"] == "learned":
                d_emb = torch.cat((d_emb,self.proj_matrix[i](z_codes[:,i,:]).unsqueeze(-2)), dim =-2)
            if self.training_params["codebook"] == "default":
                temp_emb = dacModel.quantizer.quantizers[i].decode_code(z_codes[:,i,:])
                temp_emb = dacModel.quantizer.quantizers[i].out_proj(temp_emb)
                temp_emb = einops.rearrange(temp_emb, "b d t -> b t (1) d")
                d_emb = torch.cat((d_emb,temp_emb), dim=-2)

        if self.training_params["codes"] == "flattened":
            z_codes = z_codes.flatten(1)
            z_prime_codes = z_prime_codes.flatten(1)
            d_emb = d_emb.flatten(1,2)
        else:
            d_emb = d_emb.sum(2)
        
        d_emb = einops.rearrange(d_emb, "b t d -> b d t")

        if self.training_params["input_type"] == "continuous":
            input_emb = z
        else:
            input_emb = d_emb
        
        # latent = torch.cat((torch.zeros(z.shape[0], z.shape[1],1).to(self.device), z),-1)
        latent = self.enc(input_emb)
        
        # latent = torch.cat((torch.zeros(latent.shape[0], latent.shape[1],1).to(self.device), latent),-1)
        # latent = z.swapaxes(1,2)
        # enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=9, nhead=3), num_layers=6).to(self.device)
        # latent = enc(latent)
        # latent_tuple = latent.chunk(latent.shape[-1]//8,-1)
        # latent = torch.zeros((latent.shape[0], latent.shape[1], 0)).to(self.device)
        # for l in latent_tuple:
        #     latent = torch.cat((latent,l.mean(-1, keepdim=True)), dim=-1)

        if self.training_params["c_length"]:
            c_length = self.training_params["c_length"]
        else:
            c_length = latent.shape[-1]

        #extend the length of pitch to be size of latent space
        p = einops.repeat(p, "b  -> b (t )",t=c_length)
        note_num = p_prime[0]
        p_prime = einops.repeat(p_prime, "b  -> b (t )",t=c_length) 

        #get pitch embeddings
        p_latent = einops.rearrange(self.p_enc(p), "b t d -> b d t")
        p_prime_latent = einops.rearrange(self.p_enc(p_prime), "b t d -> b d t")

        if self.training_params["reconstruction"] == "same":
            p_prime_latent = p_latent

        # rest_filt,_ = self.filter_generator(p_latent)
        # _,dec_filt = self.filter_generator(p_prime_latent)

        emb_loss = torch.tensor(0.0).to(self.device)

        # if self.training_params["decoder_type"] == "transformer":
                
        #     #make positional embeddings
        #     max_len = latent.shape[2]
        #     d_model = latent.shape[1]
        #     batch_size = latent.shape[0]
        #     pe = torch.zeros(max_len, d_model)
        #     position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        #     div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        #     pe[:, 0::2] = torch.sin(position * div_term)
        #     pe[:, 1::2] = torch.cos(position * div_term)
        #     pe = pe.transpose(0, 1).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
        #     sz = latent.shape[-1]
        #     # mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        #     # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        #     latent = latent + pe
        #     latent = einops.rearrange(latent, "b d t -> b t d")
        #     p_latent = einops.rearrange(p_latent, "b d t -> b t d")
        #     p_prime_latent = einops.rearrange(p_prime_latent, "b d t -> b t d")
        
        #get rest embedding
        if self.training_params["disentangle"] == "subtract":
            rest_emb = latent - p_latent
            self.rest_emb = rest_emb
            dec_input = rest_emb + p_prime_latent
            z_hat = self.dec(dec_input)

        if self.training_params["disentangle"] == None:
            rest_emb = latent 
            self.rest_emb = rest_emb
            dec_input = latent

            z_hat = self.dec(dec_input)
            
        
        if self.training_params["disentangle"] == "conv":
            rest_emb = self.rest_conv(torch.cat((latent,p_latent), dim=1))
            self.rest_emb = rest_emb
            dec_input = self.transform_conv(torch.cat((rest_emb,p_prime_latent), dim=1))
            z_hat = self.dec(dec_input)

        if self.training_params["disentangle"] == "conv_single":
            rest_emb = latent
            self.rest_emb = rest_emb
            dec_input = self.transform_conv(torch.cat((latent,p_prime_latent), dim=1))
            z_hat = self.dec(dec_input)

        if self.training_params["disentangle"] == "concat":
            rest_emb = latent
            self.rest_emb = rest_emb
            dec_input = torch.cat((latent,p_prime_latent), dim=1)
            z_hat = self.dec(dec_input)

        if self.training_params["disentangle"] == "pesto":
            p_prime_latent = einops.repeat(self.pesto_emb[note_num.item()].to(self.device), "t d  -> (b) d t",b=latent.shape[0])[:,:,:latent.shape[-1]]
            rest_emb = latent
            self.rest_emb = rest_emb
            dec_input = torch.cat((latent,p_prime_latent), dim=1)
            z_hat = self.dec(dec_input)

        if self.training_params["disentangle"] == "FiLM":
            film = self.film(p_prime_latent[:, :, 0]).chunk(self.n_FiLM *2, 1)
            out = latent
            rest_emb = latent
            self.rest_emb = rest_emb
            for i, resblock in enumerate(self.resblocks):
                out = resblock(out, film[i * 2], film[i * 2 + 1])
            z_hat = self.dec(out)

        if self.training_params["disentangle"] == "across_inst":

            #get latent from audio input
            d_prime_emb = torch.zeros(z.shape[0],  z.shape[-1], 0,self.input_emb_dim).to(self.device)

            for i in range(9):
                if self.training_params["codebook"] == "learned":
                    d_prime_emb = torch.cat((d_prime_emb,self.proj_matrix[i](z_prime_codes_saved[:,i,:]).unsqueeze(-2)), dim =-2)
                if self.training_params["codebook"] == "default":
                    temp_emb = dacModel.quantizer.quantizers[i].decode_code(z_prime_codes_saved[:,i,:])
                    temp_emb = dacModel.quantizer.quantizers[i].out_proj(temp_emb)
                    temp_emb = einops.rearrange(temp_emb, "b d t -> b t (1) d")
                    d_prime_emb = torch.cat((d_prime_emb,temp_emb), dim=-2)

            if self.training_params["codes"] == "flattened":
                z_codes = z_codes.flatten(1)
                z_prime_codes = z_prime_codes.flatten(1)
                d_prime_emb = d_prime_emb.flatten(1,2)
            else:
                d_prime_emb = d_prime_emb.sum(2)
            
            d_prime_emb = einops.rearrange(d_prime_emb, "b t d -> b d t")

            if self.training_params["input_type"] == "continuous":
                input_prime_emb = z_prime
            else:
                input_prime_emb = d_prime_emb
            
            t_emb = self.t_enc(input_emb)
            t_emb = einops.repeat(t_emb, "b d t  -> b d (t repeat)",repeat=latent.shape[-1]) 
            t_emb_prime = self.t_enc(input_prime_emb)
            t_emb_prime = einops.repeat(t_emb_prime, "b d t  -> b d (t repeat)",repeat=latent.shape[-1]) 

            emb_loss = MSE_loss(t_emb[:,:,0], t_emb_prime[:,:,0])

            rest_emb = latent
            self.rest_emb = rest_emb

            dec_input = torch.cat((latent,p_prime_latent, t_emb), dim=1)
            # dec_input =latent + p_prime_latent + t_emb
            z_hat = self.dec(dec_input)
            

        if self.training_params["disentangle"] == "random_inst":
            prime_emb = torch.zeros(z_prime.shape[0],  z_prime.shape[-1], 512).to(self.device)
            for i in range(9):
                prime_emb += self.proj_matrix[i](z_prime_codes[:,i,:])
            prime_emb = einops.rearrange(prime_emb, "b t d -> b d t")
            t_emb = self.t_enc(prime_emb)
            t_emb = einops.repeat(t_emb, "b d t  -> b d (t repeat)",repeat=latent.shape[-1])
            rest_emb = latent
            self.rest_emb = rest_emb

            if self.training_params["decoder_type"] == "transformer":
                t_emb = einops.rearrange(t_emb, "b d t -> b t d")

            dec_input = torch.cat((latent,p_prime_latent, t_emb), dim=1)
            z_hat = self.dec(dec_input)
        
        if self.training_params["disentangle"] == "decoder":
            rest_emb = latent 
            self.rest_emb = rest_emb
            dec_input = latent

            z_hat = self.dec(dec_input, p_prime_latent)

        if self.training_params["loss_type"] == "mse":
            loss = MSE_loss
            target = z
            target_prime = z_prime
        else:
            loss = C_loss
            target = z_codes
            target_prime = z_prime_codes

        # token_predict_loss = torch.tensor(0.0).to(self.device)
        # indices = []
        # #project to lower dimensions
        # for i in range(9):
        #     i_latent = z_hat[:, :, i, :]
        #     i_latent = dacModel.quantizer.quantizers[i].in_proj(i_latent)
        #     encodings = einops.rearrange(i_latent, "b d t -> (b t) d")
        #     targets = einops.rearrange(z_codes[:,i,:], "b t -> (b t)")
        #     codebook = dacModel.quantizer.quantizers[i].codebook.weight # codebook: (N x D)

        #     # L2 normalize encodings and codebook (ViT-VQGAN)
        #     encodings = F.normalize(encodings)
        #     codebook = F.normalize(codebook)

        #     # Compute euclidean distance with codebook
        #     dist = (
        #         encodings.pow(2).sum(1, keepdim=True)
        #         - 2 * encodings @ codebook.t()
        #         + codebook.pow(2).sum(1, keepdim=True).t()
        #     )


        #     prob = F.log_softmax(-dist,dim=1)
        #     token_predict_loss += C_loss(prob, targets).mean()
        #     code = einops.rearrange(torch.argmax(prob, dim=1), "(b t) -> b t", b=z.shape[0])
        #     indices.append(code)

        # z_hat = torch.stack(indices, dim=1)

        # z_hat = einops.rearrange(dacModel.quantizer(z_hat)[1], "b c t -> b t c")

        # emb_list = []
        # for i in range(9):
        #     e  = einops.rearrange(self.proj_matrix[i](z_hat[:,:,i]).unsqueeze(-1), "b t e c -> b e c t")
        #     e = torch.nn.functional.log_softmax(e, dim=1).contiguous()
        #     emb_list.append(e)

        # z_hat = torch.cat(emb_list, dim=2) 

        #make contiguous
        z_hat = z_hat.contiguous()

        if self.training_params["codes"] == "flattened":
            z_hat = z_hat[:,:,0,:]

        if self.training_params["input_type"] == "continuous":
            z_hat = z_hat[:,:,0,:]

        #categorical reconstruction loss
        if self.training_params["reconstruction"] == "same":
            token_predict_loss = loss(z_hat, target)
        else:
            token_predict_loss = loss(z_hat, target_prime)

        cosine_loss = torch.tensor(0.0).to(self.device)

        #cosine loss only if disentangling
        if self.training_params["disentangle"] != None:
            if self.training_params["cosine_loss"]:
                if self.training_params["disentangle"] != "concat" and  self.training_params["disentangle"] != "conv_single":
                    cosine_loss = COS_loss(p_latent.reshape(p_latent.shape[0], p_latent.shape[1]*p_latent.shape[2]), rest_emb.reshape(rest_emb.shape[0], rest_emb.shape[1]*rest_emb.shape[2]), torch.full((rest_emb.shape[0],), -1).to(self.device))
                cosine_loss += COS_loss(p_prime_latent.reshape(p_prime_latent.shape[0], p_prime_latent.shape[1]*p_prime_latent.shape[2]), rest_emb.reshape(rest_emb.shape[0], rest_emb.shape[1]*rest_emb.shape[2]), torch.full((rest_emb.shape[0],), -1).to(self.device))

        #weight the loss per hierarchical token
        if self.training_params["loss_type"] == "hierarchical_nll":
            token_predict_loss *= torch.linspace(1,0.1,9).to(self.device)[None, :, None]
            token_predict_loss = token_predict_loss.mean()

        if self.training_params["input_type"] != "continuous":
            z_hat = torch.argmax(z_hat, dim=1)
        # z_hat = dacModel.quantizer(z_hat)[1]

        if self.training_params["codes"] == "flattened":
            z_hat = z_hat.unflatten(-1, (9,-1))

        loss = {"t_predict": token_predict_loss,
                "cosine": cosine_loss,
                "emb": emb_loss}

        predict = {"z" : z_hat}

        return loss, predict

    def generate(self, dacModel, z, p, p_prime):

        #convert codes to embedding dimensions
        z_codes = z
        with torch.no_grad():
            z = dacModel.quantizer.from_codes(z)[0]

        #start token
        start_seq = torch.zeros((z.shape[0], self.emb_dim,1)).to(self.device)
        seq = torch.zeros((z.shape[0], 9 , 0)).int().to(self.device)

        p_input = p
        p_prime_input = p_prime

        # keep iterating until full audio clip is created
        while seq.shape[-1] < 344:
            if seq.shape[-1] == 0:
                latent = start_seq
            else:
                latent = dacModel.quantizer.from_codes(seq)[0]
                latent = self.enc(latent)
                latent = torch.cat((start_seq, latent),-1)

            #reinitiallize p and primes every loop
            p = p_input
            p_prime = p_prime_input

            #extend the length of pitch to be size of latent space
            p = einops.repeat(p, "b  -> b (t)",t=latent.shape[-1])
            p_prime = einops.repeat(p_prime, "b  -> b (t )",t=latent.shape[-1])

            #get pitch embeddings
            p_latent = einops.rearrange(self.p_enc(p), "b t d -> b d t")
            p_prime_latent = einops.rearrange(self.p_enc(p_prime), "b t d -> b d t")

            #get rest embedding by subtract former pitch
            rest_emb = latent - p_latent
            self.rest_emb = rest_emb

            z_hat = self.dec(rest_emb + p_prime_latent)

            #convert log softmax output to probability distribution and get last prediction
            z_hat = torch.exp(z_hat[:, :, :, -1:])
            z_hat = torch.argmax(z_hat, dim=1)

            #sample for distribution for every rvq level
            # new_tokens = torch.zeros((0,9,1)).to(self.device)
            # for j in range(z_hat.shape[0]):
            #     sampled_tokens = torch.zeros((1,0,1)).to(self.device)
            #     for i in range(9):
            #         sampled_tokens = torch.cat((sampled_tokens, torch.distributions.Categorical(z_hat[j,:,i,0]).sample().reshape(1,1,1)), dim=1)
                # new_tokens = torch.cat((new_tokens, sampled_tokens), dim=0)

            #append sequence
            seq = torch.cat((seq, new_tokens.int()), -1)

        return seq



class disentangle(nn.Module):
    def __init__(self, device=None):
        super().__init__()

        self.device = device
        self.enc_train = True

        self.pitch_emb = None
        self.rest_emb = None

        #pitch countour (pc) numerator and denominator
        self.pc_num = 3
        self.pc_denom = 4

        self.emb_dim = 32

        # model_path = dac.utils.download(model_type="44khz") 
        # self.dacModel = dac.DAC.load(model_path)
        # self.dacModel.eval()

        self.pitch_predict = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn_custom.ResidualWrapper(
            nn.Sequential(
            nn.SyncBatchNorm(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=3, dilation=3),
        )),
        nn.Conv1d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=6,  dilation=6),
        nn.SyncBatchNorm(32),
        nn.ReLU(),
        nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=9, dilation=9),
        nn.LogSoftmax(dim=1),
        # nn.ReLU(),
            )

        self.mfcc_predict = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=3, padding=1, dilation=1),
            nn_custom.ResidualWrapper(
            nn.Sequential(
            nn.SyncBatchNorm(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=3, dilation=3),
        )),
        nn.Conv1d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=6, dilation=6),
        nn.SyncBatchNorm(32),
        nn.ReLU(),
        nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=9, dilation=9),
        nn.SyncBatchNorm(32),
        nn.ReLU(),
        nn.Conv1d(in_channels=32, out_channels=20, kernel_size=3, padding=12, dilation=12),
        nn.Sigmoid(),
            )

        self.rms_predict = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=3, padding=1, dilation=1),
            nn_custom.ResidualWrapper(
            nn.Sequential(
            nn.SyncBatchNorm(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=3, dilation=3),
        )),
        nn.Conv1d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=6, dilation=6),
        nn.SyncBatchNorm(32),
        nn.ReLU(),
        nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=9, dilation=9),
        nn.SyncBatchNorm(32),
        nn.ReLU(),
        nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=12, dilation=12),
        nn.SyncBatchNorm(32),
        nn.ReLU(),
        nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, padding=15, dilation=15),
        nn.Sigmoid(),
            )

        self.encode_pitch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn_custom.ResidualWrapper(
            nn.Sequential(
            nn.SyncBatchNorm(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
            nn.ReLU(),
        )),
            nn.Conv1d(in_channels=32, out_channels=self.emb_dim, kernel_size=1, stride=1),
            nn.ReLU(),
        )

        self.encode_mfcc = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn_custom.ResidualWrapper(
            nn.Sequential(
            nn.SyncBatchNorm(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
            nn.ReLU(),
        )),
            nn.Conv1d(in_channels=32, out_channels=self.emb_dim, kernel_size=1, stride=1),
            nn.ReLU(),
        )

        self.encode_rms = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn_custom.ResidualWrapper(
            nn.Sequential(
            nn.SyncBatchNorm(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
            nn.ReLU(),
        )),
            nn.Conv1d(in_channels=32, out_channels=self.emb_dim, kernel_size=1, stride=1),
            nn.ReLU(),
        )

        self.encode_rest = nn.Sequential(
            # nn.Conv1d(in_channels=9, out_channels=32, kernel_size=5, stride=1, padding=2, dilation=1),
            # nn.SyncBatchNorm(32),
            # nn.ReLU(),
            # nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=6, dilation=3),
            # nn.SyncBatchNorm(32),
            # nn.ReLU(),
            # nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=18, dilation=9),
            # nn.SyncBatchNorm(32),
            # nn.ReLU(),
            # nn.Conv1d(in_channels=32, out_channels=32, kernel_size=6, stride=2, padding=0, dilation=1),
            # nn.SyncBatchNorm(32),
            # nn.ReLU(),
            # nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, dilation=1),
            # nn.SyncBatchNorm(32),
            # nn.ReLU(),
            # nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=6, dilation=3),
            # nn.SyncBatchNorm(32),
            # nn.ReLU(),
            # nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=18, dilation=9),
            # nn.SyncBatchNorm(32),
            # nn.ReLU(),
            # nn.Conv1d(in_channels=32, out_channels=self.emb_dim, kernel_size=6, stride=4, padding=0, dilation=1),
            # nn.ReLU(),
            nn.Conv1d(in_channels=9, out_channels=32, kernel_size=5, stride=1, dilation=1),
            # nn_custom.ResidualWrapper(
            # nn.Sequential(
            nn.SyncBatchNorm(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, dilation=3),
            nn.ReLU(),
        # )
        # ),
        nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, dilation=6),
        nn.SyncBatchNorm(32),
        nn.ReLU(),
        nn.Conv1d(in_channels=32, out_channels=self.emb_dim, kernel_size=6, stride=3, dilation=1),
        nn.ReLU(),
        nn.Conv1d(in_channels=32, out_channels=self.emb_dim, kernel_size=5, stride=1, padding=2, dilation=1),
        nn.ReLU()
            )

        # self.rest_transform =nn.Conv1d(in_channels=self.emb_dim+1, out_channels=self.emb_dim, kernel_size=5, stride=1, padding=2, dilation=1)

        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose1d(in_channels=self.emb_dim, out_channels=64, kernel_size=6, stride=4, padding=0, dilation=1),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2, dilation=1),
        #     nn.SyncBatchNorm(128),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=6, dilation=3),
        #     nn.SyncBatchNorm(256),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=18, dilation=9),
        #     nn.SyncBatchNorm(256),
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(in_channels=256, out_channels=256, kernel_size=6, stride=2, padding=0, dilation=1),
        #     nn.SyncBatchNorm(256),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2, dilation=1),
        #     nn.SyncBatchNorm(256),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=6, dilation=3),
        #     nn.SyncBatchNorm(256),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=18, dilation=9),
        #     nn.SyncBatchNorm(256),
        #     nn.ReLU(),
        #     nn.Conv1d(256, 1024*9, kernel_size=7, stride=1, padding=3, dilation=1),
        #     nn.Unflatten(1, (1024,9)),
        #     nn.LogSoftmax(dim=1),
        #     )
        self.decoder = nn.Sequential(
                nn.SyncBatchNorm(self.emb_dim),
                nn.ConvTranspose1d(self.emb_dim, 32, kernel_size=3, stride=3, padding=1, dilation=1),
                nn.ReLU(),
                nn.ReLU(),
                # nn_custom.ResidualWrapper(
                    nn.Sequential(
                    nn.SyncBatchNorm(32),
                    nn.ReLU(),
                    nn.ConvTranspose1d(32, 64, kernel_size=7, stride=1, padding=9, dilation=3),
                    nn.ReLU(),
                ),
                # ),
                # nn_custom.ResidualWrapper(
                    nn.Sequential(
                    nn.SyncBatchNorm(64),
                    nn.ReLU(),
                    nn.ConvTranspose1d(64, 128,kernel_size=7, stride=1, padding=18, dilation=6),
                    nn.ReLU(),
                ),
                # ),
                # nn_custom.ResidualWrapper(
                    nn.Sequential(
                    nn.SyncBatchNorm(128),
                    nn.ReLU(),
                    nn.ConvTranspose1d(128, 256, kernel_size=7, stride=1, padding=27, dilation=9),
                    nn.ReLU(),
                ),
                # ),
                # # nn_custom.ResidualWrapper(
                    nn.Sequential(
                    nn.SyncBatchNorm(256),
                    nn.ReLU(),
                    nn.ConvTranspose1d(256, 256,kernel_size=7, stride=1, padding=36,dilation=12),
                    nn.ReLU(),
                ),
                # ),
                nn_custom.GRUWrap(256,256,1, batch_first=True),
                # nn.ReLU(),
                # nn.ConvTranspose1d(256, 256, kernel_size=7, stride=1,  padding=3, dilation=1),
                # nn.ReLU(),
                # nn.ConvTranspose1d(256, 256, kernel_size=7, stride=1,  padding=3, dilation=1),
                # nn.ReLU(),
                nn.ConvTranspose1d(256, 256, kernel_size=7, stride=1,  padding=31, dilation=15),
                nn.ReLU(),
                # nn.ConvTranspose1d(1024, 1024*9, kernel_size=1, stride=1, padding=0, dilation=1),
                nn_custom.SwapAxes((1,2)),
                nn.Linear(256,1024*9),
                nn_custom.SwapAxes((1,2)),
                nn.Unflatten(1, (1024,9)),
                nn.LogSoftmax(dim=1),
                # nn.SyncBatchNorm(1024),
                # nn.ReLU(), 
                # nn.Sigmoid(),
            )
        # self.dacModel = nn.DataParallel(self.dacModel)
        # self.pitch_predict = nn.DataParallel(self.pitch_predict)
        # self.mfcc_predict = nn.DataParallel(self.mfcc_predict)
        # self.rms_predict = nn.DataParallel(self.rms_predict)
        # self.encode_pitch = nn.DataParallel(self.encode_pitch)
        # self.encode_mfcc = nn.DataParallel(self.encode_mfcc)
        # self.encode_rms = nn.DataParallel(self.encode_rms)
        # self.encode_rest = nn.DataParallel(self.encode_rest)
        # self.decoder = nn.DataParallel(self.decoder)

    def stop_encoder_training(self):

        self.enc_train = False

    def get_new_sample(self, z, p=None, p_start=None, mfcc=None, rms=None, inst=None):
        #get z input in coninous mode
        z_codes = z
        # z = self.dacModel.quantizer.from_codes(z)[0]

        # p = p.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
        # p[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)
        # p_embedding = self.encode_pitch(p)

        #make predicted pitch higher dimmensional for loss
        # p_embedding = torch.zeros(rest_embedding.shape).to(self.device)
        # p_embedding[:, 0,:] = torch.argmax(p, dim=1)

        #decode back to z
        # z_hat = self.decoder(p_embedding + rest_embedding)

         #Convert from probability vector to single value with argmax
        # z_hat = torch.argmax(z_hat, dim=1)

        # #predict pitch
        # p_predict = self.pitch_predict(z)
        # p_predict = torch.argmax(p_predict, dim=1).float().unsqueeze(1)

        # #predict mfcc 
        # m_predict = self.mfcc_predict(z)

        # # #predict rms
        # r_predict = self.rms_predict(z)


        # if mfcc != None:
        #     mfcc = mfcc[..., :z.shape[-1]]
        
        # if rms != None:
        #     rms = rms[..., :z.shape[-1]]

        # if p != None:
        #     p = p.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
        #     # p[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)
        #     p_embedding = self.encode_pitch(p)
        # else:
        #     p_embedding = self.encode_pitch(p_predict)

        # # if mfcc != None:
        # #     m_embedding = self.encode_mfcc(mfcc)
        # # else:
        # #     m_embedding = self.encode_mfcc(m_predict)
        # #     mfcc = m_predict

        # if rms != None:
        #     r_embedding = self.encode_rms(rms)
        # else:
        #     r_embedding = self.encode_rms(r_predict)

        # # if p != None:
        # #     p_predict = p

        # # if mfcc != None:
        # #     m_predict = mfcc

        # # if rms != None:
        # #     r_predict = rms

        # if inst != None:
        #     inst = inst.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
        #     m_embedding = self.encode_mfcc(inst)
        # else:
        #     inst = torch.tensor([759]).to(self.device)
        #     inst = i_prime.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
        #     m_embedding = self.encode_mfcc(inst)

        #get the rest embedding form the rest encoder (should learn to subtract pitch)
        rest_embedding = self.encode_rest(z.float())
        
        p_start = p_start.unsqueeze(1).expand(-1, rest_embedding.shape[-1]).unsqueeze(1).float()
        p_start_embedding = self.encode_pitch(p_start)
        p = p.unsqueeze(1).expand(-1, rest_embedding.shape[-1]).unsqueeze(1).float()
        p_embedding = self.encode_pitch(p)
        rest_embedding =  rest_embedding - p_start_embedding

        #make predicted pitch higher dimmensional for loss
        # p_embedding = torch.zeros(rest_embedding.shape).to(self.device)
        # p_embedding[:, 0,:] = torch.argmax(p_prime, dim=1)
        # p_embedding[:, 0,:] = p_prime[:,0,:]

        #put embeddings into class public class attributes for access in computing metrics

        #decode back to z
        z_hat = self.decoder(p_embedding + rest_embedding)#self.decoder(torch.cat((p_predict, m_predict, r_predict), dim=1))


        #Convert from probability vector to single value with argmax
        z_hat = torch.argmax(z_hat, dim=1)

        return z_hat

    def train_enc(self, z, p, mfcc, rms):
        #get z input in coninuous mode
        z_codes = z
        # with torch.no_grad():
        #     z = self.dacModel.quantizer.from_codes(z)[0]

        #loss categorical and regression loss functions
        C_loss = nn.NLLLoss()
        MSE_loss = nn.MSELoss()

        #predict pitch
        p_predict = self.pitch_predict(z)

        # #predict mfcc 
        m_predict = self.mfcc_predict(z)

        # #predict rms
        r_predict = self.rms_predict(z)

        z_hat = self.decoder(self.encode_rest(z))

        #get loss of predicting token
        token_predict_loss = C_loss(z_hat, z_codes)
        z_hat = torch.argmax(z_hat, dim=1)

        p = p.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
        p[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)

        mfcc = mfcc[..., :z.shape[-1]]
        rms = rms[..., :z.shape[-1]]

        pitch_predict_loss = C_loss(p_predict, p[:,0,:].long())
        mfcc_predict_loss = MSE_loss(m_predict, mfcc)
        rms_predict_loss = MSE_loss(r_predict, rms)

        loss = {"t_predict": token_predict_loss,
                "p_predict": pitch_predict_loss,
                "m_predict": mfcc_predict_loss,
                "r_predict": rms_predict_loss,
                }

        predict = { "z" : z_hat,
                    "pitch" : p_predict,
                    "mfcc" : m_predict,
                    "rms" : r_predict
        }

        return loss, predict

    def train_recon(self, z,  z_prime, p_prime,mfcc_prime, rms_prime):
        #get z input in coninuous mode
        z_codes = z
        # with torch.no_grad():
        #     z = self.dacModel.quantizer.from_codes(z)[0]
        

        #get z_prime in continuous mode
        z_prime_codes = z_prime
        with torch.no_grad():
            z_prime = self.dacModel.quantizer.from_codes(z_prime)[0]

        #loss categorical and regression loss functions
        C_loss = nn.NLLLoss()

        p_prime = p_prime.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
        p_prime[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)
        mfcc_prime = mfcc_prime[..., :z.shape[-1]]
        rms_prime = rms_prime[..., :z.shape[-1]]

        #synchronise whether encoder is used across batches
        # device = self.device#torch.device("cpu")
        # rank = torch.distributed.get_rank()
        # if rank == 0:
        #     objects = torch.tensor([random.choice([True, False]), random.choice([True, False]), random.choice([True, False])]).to(device)
        #     torch.distributed.send(objects, dst=1)
        # else:
        #     objects = torch.tensor([False, False, False]).to(device)
        #     torch.distributed.recv(objects, src=0)


        #predict pitch
        # p_predict = self.pitch_predict(z_prime)
        # p_predict = torch.argmax(p_predict, dim=1).float().unsqueeze(1)

        # # #predict mfcc 
        # m_predict = self.mfcc_predict(z_prime)

        # # #predict rms
        # r_predict = self.rms_predict(z_prime)
        
        # torch.distributed.barrier()


        #randomly select from ground truth vs predicted


        p_embedding = self.encode_pitch(p_prime)

        # m_embedding = self.encode_mfcc(mfcc_prime)

        r_embedding = self.encode_rms(rms_prime)
        

        #get the rest embedding form the rest encoder (should learn to subtract pitch)
        rest_embedding = self.encode_rest(z) 

        #decode back to z
        z_hat = self.decoder(p_embedding + r_embedding + rest_embedding)

        #get loss of predicting token
        token_predict_loss = C_loss(z_hat, z_prime_codes)

        loss = {"t_predict": token_predict_loss}

        predict = {"z" : z_hat}

        return loss,predict

    def train_autoencoder(self, z):
        #get z input in coninuous mode
        z_codes = z
        # with torch.no_grad():
        #     z = self.dacModel.quantizer.from_codes(z)[0]
        

        #loss categorical and regression loss functions
        C_loss = nn.NLLLoss()
        

        #get the rest embedding form the rest encoder (should learn to subtract pitch)
        latent = self.encode_rest(z.float()) 

        #decode back to z
        z_hat = self.decoder(latent)

        #get loss of predicting token
        token_predict_loss = C_loss(z_hat, z_codes)

        loss = {"t_predict": token_predict_loss}

        predict = {"z" : z_hat}

        return loss,predict


    # def forward(self, z, p, mfcc, rms, inst, z_prime, p_prime,mfcc_prime, rms_prime, inst_prime):
        
    #     #get z input in coninuous mode
    #     z_codes = z
    #     with torch.no_grad():
    #         z = self.dacModel.quantizer.from_codes(z)[0]
        

    #     #get z_prime in continuous mode
    #     z_prime_codes = z_prime
    #     with torch.no_grad():
    #         z_prime = self.dacModel.quantizer.from_codes(z_prime)[0]
        
    #     #loss categorical and regression loss functions
    #     C_loss = nn.NLLLoss()
    #     MSE_loss = nn.MSELoss()
    #     COS_loss = nn.CosineEmbeddingLoss()

    #     #predict pitch
    #     # p_predict = self.pitch_predict(z_prime)
    #     # p_predict = torch.argmax(p_predict, dim=1).float().unsqueeze(1)

    #     # #predict mfcc 
    #     # m_predict = self.mfcc_predict(z_prime)

    #     # #predict rms
    #     # r_predict = self.rms_predict(z_prime)
 
    #     p = p.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
    #     # p[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)
    #     p_prime = p_prime.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
    #     # p_prime[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)
    #     mfcc = mfcc[..., :z.shape[-1]]
    #     mfcc_prime = mfcc_prime[..., :z.shape[-1]]
    #     rms = rms[..., :z.shape[-1]]
    #     rms_prime = rms_prime[..., :z.shape[-1]]
    #     # inst = inst.unsqueeze(1).expand(-1, rest_embedding.shape[-1]).unsqueeze(1).float()
    #     # inst_prime = inst_prime.unsqueeze(1).expand(-1, rest_embedding.shape[-1]).unsqueeze(1).float()

    #     # recon_loss = MSE_loss(z_prime, z_prime_hat)
    #     #get loss of pitch prediction
    #     # pitch_predict_loss = C_loss(p_predict, p[:,0,:].long())
    #     # mfcc_predict_loss = MSE_loss(m_predict, mfcc)
    #     # rms_predict_loss = MSE_loss(r_predict, rms)
        

    #     # p_prime = p_prime.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
    #     # p_prime[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)

    #     #get pitch embeding from pitch encoder
    #     # p_embedding = self.encode_pitch(torch.argmax(p_prime, dim=1).float().unsqueeze(1))


    #     p_embedding = self.encode_pitch(p)
    #     # r_embedding = self.encode_rms(rms)
    #     # m_embedding = self.encode_mfcc(inst)
    #     p_embedding_prime = self.encode_pitch(p_prime)
    #     # r_embedding_prime = self.encode_rms(rms_prime)
    #     # m_embedding_prime = self.encode_mfcc(inst_prime)
        

    #     #get the rest embedding form the rest encoder (should learn to subtract pitch)

    #     rest_embedding = self.encode_rest(z)
    #     rest_embedding =  rest_embedding - p_embedding


    #     #make predicted pitch higher dimmensional for loss
    #     # p_embedding = torch.zeros(rest_embedding.shape).to(self.device)
    #     # p_embedding[:, 0,:] = torch.argmax(p_prime, dim=1)
    #     # p_embedding[:, 0,:] = p_prime[:,0,:]

    #     #put embeddings into class public class attributes for access in computing metrics
    #     self.pitch_emb = p_embedding
    #     self.rest_emb = rest_embedding

    #     #decode back to z
    #     z_hat = self.decoder(p_embedding_prime +  rest_embedding)#self.decoder(torch.cat((p_predict, m_predict, r_predict), dim=1)) 


    #     #get loss of predicting token
    #     token_predict_loss = C_loss(z_hat, z_prime_codes)

    #     #get loss of embedding similarity
    #     # pitch_ce_loss = COS_loss(p_embedding.view(z.shape[0], p_embedding.shape[1]*p_embedding.shape[2]), rest_embedding.view(z.shape[0], rest_embedding.shape[1]*rest_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))
    #     # mfcc_ce_loss = COS_loss(m_embedding.view(z.shape[0], m_embedding.shape[1]*m_embedding.shape[2]), rest_embedding.view(z.shape[0], rest_embedding.shape[1]*rest_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))
    #     # rms_ce_loss = COS_loss(r_embedding.view(z.shape[0], r_embedding.shape[1]*r_embedding.shape[2]), rest_embedding.view(z.shape[0], rest_embedding.shape[1]*rest_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))
    #     # pm_ce_loss = COS_loss(p_embedding.view(z.shape[0], p_embedding.shape[1]*p_embedding.shape[2]), m_embedding.view(z.shape[0], m_embedding.shape[1]*m_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))
    #     # pr_ce_loss = COS_loss(p_embedding.view(z.shape[0], p_embedding.shape[1]*p_embedding.shape[2]), r_embedding.view(z.shape[0], r_embedding.shape[1]*r_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))
    #     # mr_ce_loss = COS_loss(m_embedding.view(z.shape[0], m_embedding.shape[1]*m_embedding.shape[2]), r_embedding.view(z.shape[0], r_embedding.shape[1]*r_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))

    #     #Convert from probability vector to single value with argmax
    #     z_hat = torch.argmax(z_hat, dim=1)
    #     # z_hat_con = self.dacModel.quantizer.from_codes(z_hat)[0]

    #     # recon_pitch = self.pitch_predict(z_hat_con)
    #     # recon_mfcc = self.mfcc_predict(z_hat_con)
    #     # recon_rms = self.rms_predict(z_hat_con)

    #     # pitch_predict_loss = C_loss(recon_pitch, p_prime[:,0,:].long())
    #     # mfcc_predict_loss = MSE_loss(recon_mfcc, mfcc_prime)
    #     # rms_predict_loss = MSE_loss(recon_rms, rms_prime)

    #     loss = {"t_predict": token_predict_loss,
    #             # "p_recon": pitch_predict_loss,
    #             # "m_recon": mfcc_predict_loss,
    #             # "r_recon": rms_predict_loss,
    #             # "p_ce_loss": pitch_ce_loss,
    #             # "m_ce_loss": mfcc_ce_loss,
    #             # "r_ce_loss": rms_ce_loss,
    #             # "pm_ce_loss": pm_ce_loss,
    #             # "pr_ce_loss": pr_ce_loss,
    #             # "mr_ce_loss": mr_ce_loss,
    #             }

    #     predict = {"z" : z_hat,
    #                 "pitch" : p_prime, #p_predict,
    #                 "mfcc" : mfcc_prime, #m_predict,
    #                 "rms" : rms_prime, #r_predict
    #     }

    #     return loss, predict

    def forward(self, *a, **kw):
        return self.forwardV2(*a, **kw)

    def forwardV1(self, z, p, mfcc, rms, inst, z_prime, p_prime,mfcc_prime, rms_prime, inst_prime):
        
        #get z input in coninuous mode
        z_codes = z
        with torch.no_grad():
            z = self.dacModel.quantizer.from_codes(z)[0]
        

        #get z_prime in continuous mode
        z_prime_codes = z_prime
        with torch.no_grad():
            z_prime = self.dacModel.quantizer.from_codes(z_prime)[0]
        
        #loss categorical and regression loss functions
        C_loss = nn.NLLLoss()
        MSE_loss = nn.MSELoss()
        COS_loss = nn.CosineEmbeddingLoss()
 
        p = p.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
        # p[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)
        p_prime = p_prime.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
        # p_prime[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)
        mfcc = mfcc[..., :z.shape[-1]]
        mfcc_prime = mfcc_prime[..., :z.shape[-1]]
        rms = rms[..., :z.shape[-1]]
        rms_prime = rms_prime[..., :z.shape[-1]]


        p_embedding = self.encode_pitch(p)
        p_embedding_prime = self.encode_pitch(p_prime)

        

        #get the rest embedding form the rest encoder (should learn to subtract pitch)

        rest_embedding = self.encode_rest(z)
        rest_embedding =  rest_embedding - p_embedding

        #put embeddings into class public class attributes for access in computing metrics
        self.pitch_emb = p_embedding
        self.rest_emb = rest_embedding

        #decode back to z
        z_hat = self.decoder(p_embedding_prime +  rest_embedding)#self.decoder(torch.cat((p_predict, m_predict, r_predict), dim=1)) 


        #get loss of predicting token
        token_predict_loss = C_loss(z_hat, z_prime_codes)

        z_hat = torch.argmax(z_hat, dim=1)


        loss = {"t_predict": token_predict_loss,

                }

        predict = {"z" : z_hat,
                    "pitch" : p_prime, #p_predict,
                    "mfcc" : mfcc_prime, #m_predict,
                    "rms" : rms_prime, #r_predict
        }

        return loss, predict

    def forwardV2(self, z, p, mfcc, rms, inst, z_prime, p_prime,mfcc_prime, rms_prime, inst_prime):
        
        #get z input in coninuous mode
        z_codes = z
        # with torch.no_grad():
        #     z = self.dacModel.quantizer.from_codes(z)[0]
        

        #get z_prime in continuous mode
        z_prime_codes = z_prime
        # with torch.no_grad():
        #     z_prime = self.dacModel.quantizer.from_codes(z_prime)[0]
        
        #loss categorical and regression loss functions
        C_loss = nn.NLLLoss()
        MSE_loss = nn.MSELoss()
        COS_loss = nn.CosineEmbeddingLoss()

        #predict pitch
        # p_predict = self.pitch_predict(z_prime)
        # p_predict = torch.argmax(p_predict, dim=1).float().unsqueeze(1)

        # #predict mfcc 
        # m_predict = self.mfcc_predict(z_prime)

        # #predict rms
        # r_predict = self.rms_predict(z_prime)
        rest_embedding = self.encode_rest(z.float())

        p = p.unsqueeze(1).expand(-1, rest_embedding.shape[-1]).unsqueeze(1).float()
        # p[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)
        p_out = p_prime.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
        p_prime = p_prime.unsqueeze(1).expand(-1, rest_embedding.shape[-1]).unsqueeze(1).float()
        # p_prime[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)
        mfcc = mfcc[..., :z.shape[-1]]
        mfcc_out = mfcc_prime[..., :z.shape[-1]]
        mfcc_prime = mfcc_prime[..., :rest_embedding.shape[-1]]
        rms = rms[..., :z.shape[-1]]
        rms_out = rms_prime[..., :z.shape[-1]]
        rms_prime = rms_prime[..., :rest_embedding.shape[-1]]
        # inst = inst.unsqueeze(1).expand(-1, rest_embedding.shape[-1]).unsqueeze(1).float()
        # inst_prime = inst_prime.unsqueeze(1).expand(-1, rest_embedding.shape[-1]).unsqueeze(1).float()

        # recon_loss = MSE_loss(z_prime, z_prime_hat)
        #get loss of pitch prediction
        # pitch_predict_loss = C_loss(p_predict, p[:,0,:].long())
        # mfcc_predict_loss = MSE_loss(m_predict, mfcc)
        # rms_predict_loss = MSE_loss(r_predict, rms)
        

        # p_prime = p_prime.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
        # p_prime[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)

        #get pitch embeding from pitch encoder
        # p_embedding = self.encode_pitch(torch.argmax(p_prime, dim=1).float().unsqueeze(1))


        p_embedding = self.encode_pitch(p)
        # r_embedding = self.encode_rms(rms)
        # m_embedding = self.encode_mfcc(inst)
        p_embedding_prime = self.encode_pitch(p_prime)
        # r_embedding_prime = self.encode_rms(rms_prime)
        # m_embedding_prime = self.encode_mfcc(inst_prime)
        

        #get the rest embedding form the rest encoder (should learn to subtract pitch)

        rest_embedding =  rest_embedding - p_embedding


        #make predicted pitch higher dimmensional for loss
        # p_embedding = torch.zeros(rest_embedding.shape).to(self.device)
        # p_embedding[:, 0,:] = torch.argmax(p_prime, dim=1)
        # p_embedding[:, 0,:] = p_prime[:,0,:]

        #put embeddings into class public class attributes for access in computing metrics
        self.pitch_emb = p_embedding
        self.rest_emb = rest_embedding

        #decode back to z
        z_hat = self.decoder(p_embedding_prime +  rest_embedding)#self.decoder(torch.cat((p_predict, m_predict, r_predict), dim=1)) 


        #get loss of predicting token
        token_predict_loss = C_loss(z_hat, z_prime_codes)

        #get loss of embedding similarity
        # pitch_ce_loss = COS_loss(p_embedding.view(z.shape[0], p_embedding.shape[1]*p_embedding.shape[2]), rest_embedding.view(z.shape[0], rest_embedding.shape[1]*rest_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))
        # mfcc_ce_loss = COS_loss(m_embedding.view(z.shape[0], m_embedding.shape[1]*m_embedding.shape[2]), rest_embedding.view(z.shape[0], rest_embedding.shape[1]*rest_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))
        # rms_ce_loss = COS_loss(r_embedding.view(z.shape[0], r_embedding.shape[1]*r_embedding.shape[2]), rest_embedding.view(z.shape[0], rest_embedding.shape[1]*rest_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))
        # pm_ce_loss = COS_loss(p_embedding.view(z.shape[0], p_embedding.shape[1]*p_embedding.shape[2]), m_embedding.view(z.shape[0], m_embedding.shape[1]*m_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))
        # pr_ce_loss = COS_loss(p_embedding.view(z.shape[0], p_embedding.shape[1]*p_embedding.shape[2]), r_embedding.view(z.shape[0], r_embedding.shape[1]*r_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))
        # mr_ce_loss = COS_loss(m_embedding.view(z.shape[0], m_embedding.shape[1]*m_embedding.shape[2]), r_embedding.view(z.shape[0], r_embedding.shape[1]*r_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))

        #Convert from probability vector to single value with argmax
        z_hat = torch.argmax(z_hat, dim=1)
        # z_hat_con = self.dacModel.quantizer.from_codes(z_hat)[0]

        # recon_pitch = self.pitch_predict(z_hat_con)
        # recon_mfcc = self.mfcc_predict(z_hat_con)
        # recon_rms = self.rms_predict(z_hat_con)

        # pitch_predict_loss = C_loss(recon_pitch, p_prime[:,0,:].long())
        # mfcc_predict_loss = MSE_loss(recon_mfcc, mfcc_prime)
        # rms_predict_loss = MSE_loss(recon_rms, rms_prime)

        loss = {"t_predict": token_predict_loss,
                # "p_recon": pitch_predict_loss,
                # "m_recon": mfcc_predict_loss,
                # "r_recon": rms_predict_loss,
                # "p_ce_loss": pitch_ce_loss,
                # "m_ce_loss": mfcc_ce_loss,
                # "r_ce_loss": rms_ce_loss,
                # "pm_ce_loss": pm_ce_loss,
                # "pr_ce_loss": pr_ce_loss,
                # "mr_ce_loss": mr_ce_loss,
                }

        predict = {"z" : z_hat,
                    "pitch" : p_out, #p_predict,
                    "mfcc" : mfcc_out, #m_predict,
                    "rms" : rms_out, #r_predict
        }

        return loss, predict

# class disentangle(nn.Module):
#     def __init__(self, device=None):
#         super().__init__()

#         self.device = device
#         self.enc_train = True

#         self.pitch_emb = None
#         self.rest_emb = None

#         #pitch countour (pc) numerator and denominator
#         self.pc_num = 3
#         self.pc_denom = 4

#         model_path = dac.utils.download(model_type="44khz") 
#         self.dacModel = dac.DAC.load(model_path)
#         self.dacModel.eval()

#         self.pitch_predict = nn.Sequential(
#             nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1),
#             nn_custom.ResidualWrapper(
#             nn.Sequential(
#             nn.SyncBatchNorm(256),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=3, dilation=3),
#         )),
#         nn.Conv1d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=6, dilation=6),
#         nn.SyncBatchNorm(32),
#         nn.ReLU(),
#         nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=9, dilation=9),
#         nn.LogSoftmax(dim=1),
#         # nn.ReLU(),
#             )

#         self.mfcc_predict = nn.Sequential(
#             nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=3, padding=1, dilation=1),
#             nn_custom.ResidualWrapper(
#             nn.Sequential(
#             nn.SyncBatchNorm(256),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=3, dilation=3),
#         )),
#         nn.Conv1d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=6, dilation=6),
#         nn.SyncBatchNorm(32),
#         nn.ReLU(),
#         nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=9, dilation=9),
#         nn.SyncBatchNorm(32),
#         nn.ReLU(),
#         nn.Conv1d(in_channels=32, out_channels=20, kernel_size=3, padding=12, dilation=12),
#         nn.Sigmoid(),
#             )

#         self.rms_predict = nn.Sequential(
#             nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=3, padding=1, dilation=1),
#             nn_custom.ResidualWrapper(
#             nn.Sequential(
#             nn.SyncBatchNorm(256),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=3, dilation=3),
#         )),
#         nn.Conv1d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=6, dilation=6),
#         nn.SyncBatchNorm(32),
#         nn.ReLU(),
#         nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=9, dilation=9),
#         nn.SyncBatchNorm(32),
#         nn.ReLU(),
#         nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=12, dilation=12),
#         nn.SyncBatchNorm(32),
#         nn.ReLU(),
#         nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, padding=15, dilation=15),
#         nn.Sigmoid(),
#             )

#         self.encode_pitch = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
#             nn_custom.ResidualWrapper(
#             nn.Sequential(
#             nn.SyncBatchNorm(32),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
#             nn.ReLU(),
#         )),
#             nn.Conv1d(in_channels=32, out_channels=1024, kernel_size=1, stride=1),
#             nn.ReLU(),
#         )

#         self.encode_mfcc = nn.Sequential(
#             nn.Conv1d(in_channels=20, out_channels=32, kernel_size=5, stride=1, padding=2),
#             nn_custom.ResidualWrapper(
#             nn.Sequential(
#             nn.SyncBatchNorm(32),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
#             nn.ReLU(),
#         )),
#             nn.Conv1d(in_channels=32, out_channels=1024, kernel_size=1, stride=1),
#             nn.ReLU(),
#         )

#         self.encode_rms = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
#             nn_custom.ResidualWrapper(
#             nn.Sequential(
#             nn.SyncBatchNorm(32),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
#             nn.ReLU(),
#         )),
#             nn.Conv1d(in_channels=32, out_channels=1024, kernel_size=1, stride=1),
#             nn.ReLU(),
#         )

#         self.encode_rest = nn.Sequential(
#             nn.Conv1d(in_channels=1044, out_channels=256, kernel_size=5, stride=1, padding=2, dilation=1),
#             nn_custom.ResidualWrapper(
#             nn.Sequential(
#             nn.SyncBatchNorm(256),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=3, dilation=3),
#         )),
#         nn.Conv1d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=6, dilation=6),
#         nn.SyncBatchNorm(32),
#         nn.ReLU(),
#         nn.Conv1d(in_channels=32, out_channels=1024, kernel_size=3, stride=1, padding=9, dilation=9),
#         nn.ReLU()
#             )

#         self.decoder = nn.Sequential(
#                 nn.SyncBatchNorm(1024),
#                 nn.ConvTranspose1d(1024, 256, 3, 1, 1, dilation=1),
#                 nn_custom.ResidualWrapper(
#                     nn.Sequential(
#                     nn.SyncBatchNorm(256),
#                     nn.ReLU(),
#                     nn.ConvTranspose1d(256, 256, 7, 1, 9, dilation=3),
#                     nn.ReLU(),
#                 )
#                 ),
#                 nn_custom.ResidualWrapper(
#                     nn.Sequential(
#                     nn.SyncBatchNorm(256),
#                     nn.ReLU(),
#                     nn.ConvTranspose1d(256, 256,7, 1, 18, dilation=6),
#                     nn.ReLU(),
#                 )
#                 ),
#                 # nn_custom.ResidualWrapper(
#                 #     nn.Sequential(
#                 #     nn.SyncBatchNorm(256),
#                 #     nn.ReLU(),
#                 #     nn.ConvTranspose1d(256, 512,7, 1, 27, dilation=9),
#                 #     nn.ReLU(),
#                 # ),
#                 # ),
#                 # nn_custom.ResidualWrapper(
#                     # nn.Sequential(
#                     # nn.SyncBatchNorm(512),
#                     # nn.ReLU(),
#                     # nn.ConvTranspose1d(512, 1024,7, 1, 36, dilation=12),
#                     # nn.ReLU(),
#                 # ),
#                 # ),
#                 nn_custom.GRUWrap(256,256,1, batch_first=True),
#                 nn.ReLU(),
#                 nn.ConvTranspose1d(256, 1024*9, kernel_size=7, stride=1, padding=27, dilation=9),
#                 nn.Unflatten(1, (1024,9)),
#                 nn.LogSoftmax(dim=1),
#                 # nn.SyncBatchNorm(1024),
#                 # nn.ReLU(), 
#                 # nn.Sigmoid(),
#             )
#         # self.dacModel = nn.DataParallel(self.dacModel)
#         # self.pitch_predict = nn.DataParallel(self.pitch_predict)
#         # self.mfcc_predict = nn.DataParallel(self.mfcc_predict)
#         # self.rms_predict = nn.DataParallel(self.rms_predict)
#         # self.encode_pitch = nn.DataParallel(self.encode_pitch)
#         # self.encode_mfcc = nn.DataParallel(self.encode_mfcc)
#         # self.encode_rms = nn.DataParallel(self.encode_rms)
#         # self.encode_rest = nn.DataParallel(self.encode_rest)
#         # self.decoder = nn.DataParallel(self.decoder)

#     def stop_encoder_training(self):

#         self.enc_train = False

#     def get_new_sample(self, z,p=None, mfcc=None, rms=None):
#         #get z input in coninous mode
#         z_codes = z
#         z = self.dacModel.quantizer.from_codes(z)[0]

#         # p = p.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
#         # p[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)
#         # p_embedding = self.encode_pitch(p)

#         #make predicted pitch higher dimmensional for loss
#         # p_embedding = torch.zeros(rest_embedding.shape).to(self.device)
#         # p_embedding[:, 0,:] = torch.argmax(p, dim=1)

#         #decode back to z
#         # z_hat = self.decoder(p_embedding + rest_embedding)

#          #Convert from probability vector to single value with argmax
#         # z_hat = torch.argmax(z_hat, dim=1)

#         #predict pitch
#         p_predict = self.pitch_predict(z)
#         p_predict = torch.argmax(p_predict, dim=1).float().unsqueeze(1)

#         #predict mfcc 
#         m_predict = self.mfcc_predict(z)

#         # #predict rms
#         r_predict = self.rms_predict(z)


#         if mfcc != None:
#             mfcc = mfcc[..., :z.shape[-1]]
        
#         if rms != None:
#             rms = rms[..., :z.shape[-1]]

#         if p != None:
#             p = p.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
#             p[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)
#             p_embedding = self.encode_pitch(p)
#         else:
#             p_embedding = self.encode_pitch(p_predict)

#         if mfcc != None:
#             m_embedding = self.encode_mfcc(mfcc)
#         else:
#             m_embedding = self.encode_mfcc(m_predict)
#             mfcc = m_predict

#         if rms != None:
#             r_embedding = self.encode_rms(rms)
#         else:
#             r_embedding = self.encode_rms(r_predict)

#         # if p != None:
#         #     p_predict = p

#         # if mfcc != None:
#         #     m_predict = mfcc

#         # if rms != None:
#         #     r_predict = rms

        

#         #get the rest embedding form the rest encoder (should learn to subtract pitch)
#         rest_embedding = self.encode_rest(z)

#         #make predicted pitch higher dimmensional for loss
#         # p_embedding = torch.zeros(rest_embedding.shape).to(self.device)
#         # p_embedding[:, 0,:] = torch.argmax(p_prime, dim=1)
#         # p_embedding[:, 0,:] = p_prime[:,0,:]

#         #put embeddings into class public class attributes for access in computing metrics

#         #decode back to z
#         z_hat = self.decoder(p_embedding + r_embedding + rest_embedding)#self.decoder(torch.cat((p_predict, m_predict, r_predict), dim=1))


#         #Convert from probability vector to single value with argmax
#         z_hat = torch.argmax(z_hat, dim=1)

#         return z_hat

#     def forward(self, z, p, mfcc, rms, z_prime):
        
#         #get z input in coninuous mode
#         z_codes = z
#         with torch.no_grad():
#             z = self.dacModel.quantizer.from_codes(z)[0]
        

#         #get z_prime in continuous mode
#         z_prime_codes = z_prime
#         with torch.no_grad():
#             z_prime = self.dacModel.quantizer.from_codes(z_prime)[0]
        
#         #loss categorical and regression loss functions
#         C_loss = nn.NLLLoss()
#         MSE_loss = nn.MSELoss()
#         COS_loss = nn.CosineEmbeddingLoss()

#         #predict pitch
#         p_predict = self.pitch_predict(z)

#         # #predict mfcc 
#         m_predict = self.mfcc_predict(z)

#         # #predict rms
#         r_predict = self.rms_predict(z)

#         p = p.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
#         p[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)

#         # p_prime = p_prime.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
#         # p_prime[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)

#         mfcc = mfcc[..., :z.shape[-1]]
#         rms = rms[..., :z.shape[-1]]
#         # mfcc_prime = mfcc_prime[..., :z.shape[-1]]
#         # rms_prime = rms_prime[..., :z.shape[-1]]

#         # recon_loss = MSE_loss(z_prime, z_prime_hat)
#         #get loss of pitch prediction
#         pitch_predict_loss = C_loss(p_predict, p[:,0,:].long())
#         mfcc_predict_loss = MSE_loss(m_predict, mfcc)
#         rms_predict_loss = MSE_loss(r_predict, rms)


#         p_predict = torch.argmax(p_predict, dim=1).float().unsqueeze(1)
        

#         # p_prime = p_prime.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
#         # p_prime[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)

#         #get pitch embeding from pitch encoder
#         # p_embedding = self.encode_pitch(torch.argmax(p_prime, dim=1).float().unsqueeze(1))
#         p_embedding = self.encode_pitch(p_predict)

#         m_embedding = self.encode_mfcc(m_predict)

#         r_embedding = self.encode_rms(r_predict)
        

#         #get the rest embedding form the rest encoder (should learn to subtract pitch)
#         rest_embedding = self.encode_rest(z_prime)

#         #make predicted pitch higher dimmensional for loss
#         # p_embedding = torch.zeros(rest_embedding.shape).to(self.device)
#         # p_embedding[:, 0,:] = torch.argmax(p_prime, dim=1)
#         # p_embedding[:, 0,:] = p_prime[:,0,:]

#         #put embeddings into class public class attributes for access in computing metrics
#         # self.pitch_emb = p_embedding
#         # self.rest_emb = rest_embedding

#         #decode back to z
#         z_hat = self.decoder(p_embedding + r_embedding + rest_embedding)#self.decoder(torch.cat((p_predict, m_predict, r_predict), dim=1)) 

#         #get loss of predicting token
#         token_predict_loss = C_loss(z_hat, z_codes)

#         #get loss of embedding similarity
#         pitch_ce_loss = COS_loss(p_embedding.view(z.shape[0], p_embedding.shape[1]*p_embedding.shape[2]), rest_embedding.view(z.shape[0], rest_embedding.shape[1]*rest_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))
#         # mfcc_ce_loss = COS_loss(m_embedding.view(z.shape[0], m_embedding.shape[1]*m_embedding.shape[2]), rest_embedding.view(z.shape[0], rest_embedding.shape[1]*rest_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))
#         rms_ce_loss = COS_loss(r_embedding.view(z.shape[0], r_embedding.shape[1]*r_embedding.shape[2]), rest_embedding.view(z.shape[0], rest_embedding.shape[1]*rest_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))
#         # pm_ce_loss = COS_loss(p_embedding.view(z.shape[0], p_embedding.shape[1]*p_embedding.shape[2]), m_embedding.view(z.shape[0], m_embedding.shape[1]*m_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))
#         pr_ce_loss = COS_loss(p_embedding.view(z.shape[0], p_embedding.shape[1]*p_embedding.shape[2]), r_embedding.view(z.shape[0], r_embedding.shape[1]*r_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))
#         # mr_ce_loss = COS_loss(m_embedding.view(z.shape[0], m_embedding.shape[1]*m_embedding.shape[2]), r_embedding.view(z.shape[0], r_embedding.shape[1]*r_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))

#         #Convert from probability vector to single value with argmax
#         z_hat = torch.argmax(z_hat, dim=1)

#         loss = {"t_predict": token_predict_loss,
#                 "p_predict": pitch_predict_loss,
#                 "m_predict": mfcc_predict_loss,
#                 "r_predict": rms_predict_loss,
#                 "p_ce_loss": pitch_ce_loss,
#                 # "m_ce_loss": mfcc_ce_loss,
#                 "r_ce_loss": rms_ce_loss,
#                 # "pm_ce_loss": pm_ce_loss,
#                 "pr_ce_loss": pr_ce_loss,
#                 # "mr_ce_loss": mr_ce_loss,
#                 }

#         predict = {"z" : z_hat,
#                     "pitch" : p_predict,
#                     "mfcc" : m_predict,
#                     "rms" : r_predict
#         }

#         return loss, predict

# class disentangle(nn.Module):
#     def __init__(self, device=None):
#         super().__init__()

#         self.device = device
#         self.enc_train = True

#         self.pitch_emb = None
#         self.rest_emb = None

#         #pitch countour (pc) numerator and denominator
#         self.pc_num = 3
#         self.pc_denom = 4

#         model_path = dac.utils.download(model_type="44khz") 
#         self.dacModel = dac.DAC.load(model_path)
#         self.dacModel.eval()

#         self.pitch_predict = nn.Sequential(
#             nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1),
#             nn_custom.ResidualWrapper(
#             nn.Sequential(
#             nn.SyncBatchNorm(256),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=3, dilation=3),
#         )),
#         nn.Conv1d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=6, dilation=6),
#         nn.SyncBatchNorm(32),
#         nn.ReLU(),
#         nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=9, dilation=9),
#         nn.LogSoftmax(dim=1),
#         # nn.ReLU(),
#             )

#         self.mfcc_predict = nn.Sequential(
#             nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=3, padding=1, dilation=1),
#             nn_custom.ResidualWrapper(
#             nn.Sequential(
#             nn.SyncBatchNorm(256),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=3, dilation=3),
#         )),
#         nn.Conv1d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=6, dilation=6),
#         nn.SyncBatchNorm(32),
#         nn.ReLU(),
#         nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=9, dilation=9),
#         nn.SyncBatchNorm(32),
#         nn.ReLU(),
#         nn.Conv1d(in_channels=32, out_channels=20, kernel_size=3, padding=12, dilation=12),
#         nn.Sigmoid(),
#             )

#         self.rms_predict = nn.Sequential(
#             nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=3, padding=1, dilation=1),
#             nn_custom.ResidualWrapper(
#             nn.Sequential(
#             nn.SyncBatchNorm(256),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=3, dilation=3),
#         )),
#         nn.Conv1d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=6, dilation=6),
#         nn.SyncBatchNorm(32),
#         nn.ReLU(),
#         nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=9, dilation=9),
#         nn.SyncBatchNorm(32),
#         nn.ReLU(),
#         nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=12, dilation=12),
#         nn.SyncBatchNorm(32),
#         nn.ReLU(),
#         nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, padding=15, dilation=15),
#         nn.Sigmoid(),
#             )

#         self.encode_pitch = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
#             nn_custom.ResidualWrapper(
#             nn.Sequential(
#             nn.SyncBatchNorm(32),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
#             nn.ReLU(),
#         )),
#             nn.Conv1d(in_channels=32, out_channels=1024, kernel_size=1, stride=1),
#             nn.ReLU(),
#         )

#         self.encode_mfcc = nn.Sequential(
#             nn.Conv1d(in_channels=20, out_channels=32, kernel_size=5, stride=1, padding=2),
#             nn_custom.ResidualWrapper(
#             nn.Sequential(
#             nn.SyncBatchNorm(32),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
#             nn.ReLU(),
#         )),
#             nn.Conv1d(in_channels=32, out_channels=1024, kernel_size=1, stride=1),
#             nn.ReLU(),
#         )

#         self.encode_rms = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
#             nn_custom.ResidualWrapper(
#             nn.Sequential(
#             nn.SyncBatchNorm(32),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
#             nn.ReLU(),
#         )),
#             nn.Conv1d(in_channels=32, out_channels=1024, kernel_size=1, stride=1),
#             nn.ReLU(),
#         )

#         self.encode_rest = nn.Sequential(
#             nn.Conv1d(in_channels=1044, out_channels=256, kernel_size=5, stride=1, padding=2, dilation=1),
#             nn_custom.ResidualWrapper(
#             nn.Sequential(
#             nn.SyncBatchNorm(256),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=3, dilation=3),
#         )),
#         nn.Conv1d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=6, dilation=6),
#         nn.SyncBatchNorm(32),
#         nn.ReLU(),
#         nn.Conv1d(in_channels=32, out_channels=1024, kernel_size=3, stride=1, padding=9, dilation=9),
#         nn.ReLU()
#             )

#         self.decoder = nn.Sequential(
#                 nn.SyncBatchNorm(1024),
#                 nn.ConvTranspose1d(1024, 256, 3, 1, 1, dilation=1),
#                 nn_custom.ResidualWrapper(
#                     nn.Sequential(
#                     nn.SyncBatchNorm(256),
#                     nn.ReLU(),
#                     nn.ConvTranspose1d(256, 256, 7, 1, 9, dilation=3),
#                     nn.ReLU(),
#                 )
#                 ),
#                 nn_custom.ResidualWrapper(
#                     nn.Sequential(
#                     nn.SyncBatchNorm(256),
#                     nn.ReLU(),
#                     nn.ConvTranspose1d(256, 256,7, 1, 18, dilation=6),
#                     nn.ReLU(),
#                 )
#                 ),
#                 # nn_custom.ResidualWrapper(
#                     nn.Sequential(
#                     nn.SyncBatchNorm(256),
#                     nn.ReLU(),
#                     nn.ConvTranspose1d(256, 512,7, 1, 27, dilation=9),
#                     nn.ReLU(),
#                 ),
#                 # ),
#                 # nn_custom.ResidualWrapper(
#                     nn.Sequential(
#                     nn.SyncBatchNorm(512),
#                     nn.ReLU(),
#                     nn.ConvTranspose1d(512, 1024,7, 1, 36, dilation=12),
#                     nn.ReLU(),
#                 ),
#                 # ),
#                 nn_custom.GRUWrap(1024,1024,1, batch_first=True),
#                 nn.ReLU(),
#                 nn.ConvTranspose1d(1024, 1024*9, kernel_size=7, stride=1, padding=45, dilation=15),
#                 nn.Unflatten(1, (1024,9)),
#                 nn.LogSoftmax(dim=1),
#                 # nn.SyncBatchNorm(1024),
#                 # nn.ReLU(), 
#                 # nn.Sigmoid(),
#             )
#         # self.dacModel = nn.DataParallel(self.dacModel)
#         # self.pitch_predict = nn.DataParallel(self.pitch_predict)
#         # self.mfcc_predict = nn.DataParallel(self.mfcc_predict)
#         # self.rms_predict = nn.DataParallel(self.rms_predict)
#         # self.encode_pitch = nn.DataParallel(self.encode_pitch)
#         # self.encode_mfcc = nn.DataParallel(self.encode_mfcc)
#         # self.encode_rms = nn.DataParallel(self.encode_rms)
#         # self.encode_rest = nn.DataParallel(self.encode_rest)
#         # self.decoder = nn.DataParallel(self.decoder)

#     def stop_encoder_training(self):

#         self.enc_train = False

#     def get_new_sample(self, z,p=None, mfcc=None, rms=None):
#         #get z input in coninous mode
#         z_codes = z
#         z = self.dacModel.quantizer.from_codes(z)[0]

#         # p = p.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
#         # p[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)
#         # p_embedding = self.encode_pitch(p)

#         #make predicted pitch higher dimmensional for loss
#         # p_embedding = torch.zeros(rest_embedding.shape).to(self.device)
#         # p_embedding[:, 0,:] = torch.argmax(p, dim=1)

#         #decode back to z
#         # z_hat = self.decoder(p_embedding + rest_embedding)

#          #Convert from probability vector to single value with argmax
#         # z_hat = torch.argmax(z_hat, dim=1)

#         #predict pitch
#         p_predict = self.pitch_predict(z)
#         p_predict = torch.argmax(p_predict, dim=1).float().unsqueeze(1)

#         #predict mfcc 
#         m_predict = self.mfcc_predict(z)

#         # #predict rms
#         r_predict = self.rms_predict(z)


#         if mfcc != None:
#             mfcc = mfcc[..., :z.shape[-1]]
        
#         if rms != None:
#             rms = rms[..., :z.shape[-1]]

#         if p != None:
#             p = p.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
#             p[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)
#             p_embedding = self.encode_pitch(p)
#         else:
#             p_embedding = self.encode_pitch(p_predict)

#         if mfcc != None:
#             m_embedding = self.encode_mfcc(mfcc)
#         else:
#             m_embedding = self.encode_mfcc(m_predict)
#             mfcc = m_predict

#         if rms != None:
#             r_embedding = self.encode_rms(rms)
#         else:
#             r_embedding = self.encode_rms(r_predict)

#         # if p != None:
#         #     p_predict = p

#         # if mfcc != None:
#         #     m_predict = mfcc

#         # if rms != None:
#         #     r_predict = rms

        

#         #get the rest embedding form the rest encoder (should learn to subtract pitch)
#         rest_embedding = self.encode_rest(torch.cat((z, mfcc), dim=1))

#         #make predicted pitch higher dimmensional for loss
#         # p_embedding = torch.zeros(rest_embedding.shape).to(self.device)
#         # p_embedding[:, 0,:] = torch.argmax(p_prime, dim=1)
#         # p_embedding[:, 0,:] = p_prime[:,0,:]

#         #put embeddings into class public class attributes for access in computing metrics

#         #decode back to z
#         z_hat = self.decoder(p_embedding + r_embedding + rest_embedding)#self.decoder(torch.cat((p_predict, m_predict, r_predict), dim=1))


#         #Convert from probability vector to single value with argmax
#         z_hat = torch.argmax(z_hat, dim=1)

#         return z_hat

#     def forward(self, z, p, mfcc, rms, z_prime, p_prime,mfcc_prime, rms_prime):
        
#         #get z input in coninuous mode
#         z_codes = z
#         with torch.no_grad():
#             z = self.dacModel.quantizer.from_codes(z)[0]
        

#         #get z_prime in continuous mode
#         z_prime_codes = z_prime
#         with torch.no_grad():
#             z_prime = self.dacModel.quantizer.from_codes(z_prime)[0]
        
#         #loss categorical and regression loss functions
#         C_loss = nn.NLLLoss()
#         MSE_loss = nn.MSELoss()
#         COS_loss = nn.CosineEmbeddingLoss()

#         #predict pitch
#         p_predict = self.pitch_predict(z)

#         # #predict mfcc 
#         m_predict = self.mfcc_predict(z)

#         # #predict rms
#         r_predict = self.rms_predict(z)

#         p = p.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
#         p[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)

#         p_prime = p_prime.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
#         p_prime[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)

#         mfcc = mfcc[..., :z.shape[-1]]
#         mfcc_prime = mfcc_prime[..., :z.shape[-1]]
#         rms = rms[..., :z.shape[-1]]
#         rms_prime = rms_prime[..., :z.shape[-1]]

#         # recon_loss = MSE_loss(z_prime, z_prime_hat)
#         #get loss of pitch prediction
#         pitch_predict_loss = C_loss(p_predict, p[:,0,:].long())
#         mfcc_predict_loss = MSE_loss(m_predict, mfcc)
#         rms_predict_loss = MSE_loss(r_predict, rms)


#         p_predict = torch.argmax(p_predict, dim=1).float().unsqueeze(1)
        

#         # p_prime = p_prime.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
#         # p_prime[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)

#         #get pitch embeding from pitch encoder
#         # p_embedding = self.encode_pitch(torch.argmax(p_prime, dim=1).float().unsqueeze(1))
#         p_embedding = self.encode_pitch(p_prime)

#         m_embedding = self.encode_mfcc(mfcc_prime)

#         r_embedding = self.encode_rms(rms_prime)
        

#         #get the rest embedding form the rest encoder (should learn to subtract pitch)
#         rest_embedding = self.encode_rest(torch.cat((z, mfcc), dim=1))

#         #make predicted pitch higher dimmensional for loss
#         # p_embedding = torch.zeros(rest_embedding.shape).to(self.device)
#         # p_embedding[:, 0,:] = torch.argmax(p_prime, dim=1)
#         # p_embedding[:, 0,:] = p_prime[:,0,:]

#         #put embeddings into class public class attributes for access in computing metrics
#         # self.pitch_emb = p_embedding
#         # self.rest_emb = rest_embedding

#         #decode back to z
#         z_hat = self.decoder(p_embedding + r_embedding + rest_embedding)#self.decoder(torch.cat((p_predict, m_predict, r_predict), dim=1)) 

#         #get loss of predicting token
#         token_predict_loss = C_loss(z_hat, z_prime_codes)

#         #get loss of embedding similarity
#         pitch_ce_loss = COS_loss(p_embedding.view(z.shape[0], p_embedding.shape[1]*p_embedding.shape[2]), rest_embedding.view(z.shape[0], rest_embedding.shape[1]*rest_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))
#         # mfcc_ce_loss = COS_loss(m_embedding.view(z.shape[0], m_embedding.shape[1]*m_embedding.shape[2]), rest_embedding.view(z.shape[0], rest_embedding.shape[1]*rest_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))
#         rms_ce_loss = COS_loss(r_embedding.view(z.shape[0], r_embedding.shape[1]*r_embedding.shape[2]), rest_embedding.view(z.shape[0], rest_embedding.shape[1]*rest_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))
#         # pm_ce_loss = COS_loss(p_embedding.view(z.shape[0], p_embedding.shape[1]*p_embedding.shape[2]), m_embedding.view(z.shape[0], m_embedding.shape[1]*m_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))
#         pr_ce_loss = COS_loss(p_embedding.view(z.shape[0], p_embedding.shape[1]*p_embedding.shape[2]), r_embedding.view(z.shape[0], r_embedding.shape[1]*r_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))
#         # mr_ce_loss = COS_loss(m_embedding.view(z.shape[0], m_embedding.shape[1]*m_embedding.shape[2]), r_embedding.view(z.shape[0], r_embedding.shape[1]*r_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))

#         #Convert from probability vector to single value with argmax
#         z_hat = torch.argmax(z_hat, dim=1)

#         loss = {"t_predict": token_predict_loss,
#                 "p_predict": pitch_predict_loss,
#                 "m_predict": mfcc_predict_loss,
#                 "r_predict": rms_predict_loss,
#                 "p_ce_loss": pitch_ce_loss,
#                 # "m_ce_loss": mfcc_ce_loss,
#                 "r_ce_loss": rms_ce_loss,
#                 # "pm_ce_loss": pm_ce_loss,
#                 "pr_ce_loss": pr_ce_loss,
#                 # "mr_ce_loss": mr_ce_loss,
#                 }

#         predict = {"z" : z_hat,
#                     "pitch" : p_predict,
#                     "mfcc" : m_predict,
#                     "rms" : r_predict
#         }

#         return loss, predict


class style_enc(nn.Module):
    def __init__(self, output_size=1):
        super().__init__()
        self.style_encoder_1d = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=5, stride=1, padding=2),
            nn_custom.ResidualWrapper(
            nn.Sequential(
            nn.SyncBatchNorm(1024),
            nn.ReLU(),
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1),
        )),
        nn.SyncBatchNorm(1024),
        nn.ReLU(),
        nn.Conv1d(in_channels=1024, out_channels=output_size, kernel_size=1, stride=1),
        # nn.ReLU(),
        nn.Sigmoid()
            )

        # self.style_encoder_rnn = nn.GRU(input_size=1024, hidden_size=1024, batch_first=True)

    def encode_style(self, input, length):
        encoded = self.style_encoder_1d(input)

        # # Mask positions corresponding to padding
        # length = (length // (input.shape[2] / encoded.shape[2])).to(torch.int)
        # mask = (torch.arange(encoded.shape[2], device=encoded.device) < length[:, None])[:, None, :]
        # encoded *= mask
    
        

        # if self.style_encoder_rnn is not None:
        #     encoded = encoded.transpose(1, 2)

        #     encoded = nn.utils.rnn.pack_padded_sequence(
        #         encoded, length.clamp(min=1),
        #         batch_first=True, enforce_sorted=False)
        #     print(encoded.shape)
        #     _, encoded = self.style_encoder_rnn(encoded)

        #     # Get rid of layer dimension
        #     encoded = encoded.transpose(0, 1).reshape(input.shape[0], -1)
        # else:
        #     # Compute the Gram matrix, normalized by the length squared
        #     encoded /= mask.sum(dim=2, keepdim=True) + torch.finfo(encoded.dtype).eps
        #     encoded = torch.matmul(encoded, encoded.transpose(1, 2))
        # encoded = encoded.reshape(encoded.shape[0], -1)

        return encoded, {}

    def forward(self, input_e):
        encoded_s, losses_s = self.encode_style(input_e, torch.tensor([input_e.shape[2]]))
        return encoded_s
        
class content_enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.content_encoder = nn.Sequential(
        nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=5, stride=1, padding=2),
        # nn.SyncBatchNorm(1024),
        # nn.LeakyReLU(negative_slope=0.01),
        # nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=4, stride=2, padding=2),
        nn_custom.ResidualWrapper(
            nn.Sequential(
            nn.SyncBatchNorm(1024),
            nn.ReLU(),
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1),
        )
        ),
        nn.SyncBatchNorm(1024),
        )
        self.vq = vq_custom.VQEmbedding(2048, 1024, axis=1)

    def encode_content(self, input):
        encoded = self.content_encoder(input)
        if self.vq is None:
            return encoded, encoded, {}
        return encoded #self.vq(encoded) 
    
    def forward(self, input_e):
        # encoded_c, _, losses_c = self.encode_content(input_e)
        return self.encode_content(input_e), {"commitment": torch.tensor(0), "codebook": torch.tensor(0)} #encoded_c, _, losses_c

class decoder(nn.Module):
    def __init__(self, num_features=1024):
        super().__init__()
        self.decoder = nn.Sequential(
                nn.SyncBatchNorm(num_features),
                nn.ConvTranspose1d(num_features, 1024, 1, 1),
                nn_custom.ResidualWrapper(
                    nn.Sequential(
                    nn.SyncBatchNorm(1024),
                    nn.ReLU(),
                )
                ),
                nn.ReLU(),
                nn.ConvTranspose1d(1024, 1024, kernel_size=5, stride=1, padding=2),
                nn.SyncBatchNorm(1024),
                nn.ReLU(),
            )
        
    
    def decode(self, emb):
        return self.decoder(emb)

    def forward(self, input_e):
        decoded = self.decode(input_e)
        return decoded


class ResidualStack(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        # See Section 4.1 of "Neural Discrete Representation Learning".
        layers = []
        for i in range(num_residual_layers):
            layers.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv1d(
                        in_channels=num_hiddens,
                        out_channels=num_residual_hiddens,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(),
                    nn.Conv1d(
                        in_channels=num_residual_hiddens,
                        out_channels=num_hiddens,
                        kernel_size=1,
                    ),
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = h + layer(h)

        # ResNet V1-style.
        return torch.relu(h)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_downsampling_layers,
        num_residual_layers,
        num_residual_hiddens,
    ):
        super().__init__()
        # See Section 4.1 of "Neural Discrete Representation Learning".
        # The last ReLU from the Sonnet example is omitted because ResidualStack starts
        # off with a ReLU.
        conv = nn.Sequential()
        for downsampling_layer in range(num_downsampling_layers):
            if downsampling_layer == 0:
                out_channels = num_hiddens // 2
            elif downsampling_layer == 1:
                (in_channels, out_channels) = (num_hiddens // 2, num_hiddens)

            else:
                (in_channels, out_channels) = (num_hiddens, num_hiddens)

            conv.add_module(
                f"down{downsampling_layer}",
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
            )
            conv.add_module(f"relu{downsampling_layer}", nn.ReLU())

        conv.add_module(
            "final_conv",
            nn.Conv1d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=3,
                padding=1,
            ),
        )
        self.conv = conv
        self.residual_stack = ResidualStack(
            num_hiddens, num_residual_layers, num_residual_hiddens
        )

    def forward(self, x):
        h = self.conv(x)
        return self.residual_stack(h)


class Decoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_hiddens,
        num_upsampling_layers,
        num_residual_layers,
        num_residual_hiddens,
    ):
        super().__init__()
        # See Section 4.1 of "Neural Discrete Representation Learning".
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_hiddens,
            kernel_size=3,
            padding=1,
        )
        self.residual_stack = ResidualStack(
            num_hiddens, num_residual_layers, num_residual_hiddens
        )
        upconv = nn.Sequential()
        for upsampling_layer in range(num_upsampling_layers):
            if upsampling_layer < num_upsampling_layers - 2:
                (in_channels, out_channels) = (num_hiddens, num_hiddens)

            elif upsampling_layer == num_upsampling_layers - 2:
                (in_channels, out_channels) = (num_hiddens, num_hiddens // 2)

            else:
                (in_channels, out_channels) = (num_hiddens // 2, 1)

            upconv.add_module(
                f"up{upsampling_layer}",
                nn.ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
            )
            if upsampling_layer < num_upsampling_layers - 1:
                upconv.add_module(f"relu{upsampling_layer}", nn.ReLU())

        self.upconv = upconv

    def forward(self, x):
        h = self.conv(x)
        h = self.residual_stack(h)
        x_recon = self.upconv(h)
        return x_recon


class SonnetExponentialMovingAverage(nn.Module):
    # See: https://github.com/deepmind/sonnet/blob/5cbfdc356962d9b6198d5b63f0826a80acfdf35b/sonnet/src/moving_averages.py#L25.
    # They do *not* use the exponential moving average updates described in Appendix A.1
    # of "Neural Discrete Representation Learning".
    def __init__(self, decay, shape):
        super().__init__()
        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros(*shape))
        self.register_buffer("average", torch.zeros(*shape))

    def update(self, value):
        self.counter += 1
        with torch.no_grad():
            self.hidden -= (self.hidden - value) * (1 - self.decay)
            self.average = self.hidden / (1 - self.decay ** self.counter)

    def __call__(self, value):
        self.update(value)
        return self.average


class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, use_ema, decay, epsilon):
        super().__init__()
        # See Section 3 of "Neural Discrete Representation Learning" and:
        # https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L142.

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.use_ema = use_ema
        # Weight for the exponential moving average.
        self.decay = decay
        # Small constant to avoid numerical instability in embedding updates.
        self.epsilon = epsilon

        # Dictionary embeddings.
        limit = 3 ** 0.5
        e_i_ts = torch.FloatTensor(embedding_dim, num_embeddings).uniform_(
            -limit, limit
        )
        if use_ema:
            self.register_buffer("e_i_ts", e_i_ts)
        else:
            self.register_parameter("e_i_ts", nn.Parameter(e_i_ts))

        # Exponential moving average of the cluster counts.
        self.N_i_ts = SonnetExponentialMovingAverage(decay, (num_embeddings,))
        # Exponential moving average of the embeddings.
        self.m_i_ts = SonnetExponentialMovingAverage(decay, e_i_ts.shape)

    def forward(self, x):
        flat_x = x.permute(1, 0).reshape(-1, self.embedding_dim)

        distances = (
            (flat_x ** 2).sum(1, keepdim=True)
            - 2 * flat_x @ self.e_i_ts
            + (self.e_i_ts ** 2).sum(0, keepdim=True)
        )

        encoding_indices = distances.argmin(1)


        quantized_x = F.embedding(
            encoding_indices, self.e_i_ts.transpose(0, 1)
        ).permute(1, 0,)

        # See second term of Equation (3).
        if not self.use_ema:
            dictionary_loss = ((x.detach() - quantized_x) ** 2).mean()
        else:
            dictionary_loss = None

        # See third term of Equation (3).
        commitment_loss = ((x - quantized_x.detach()) ** 2).mean()
        # Straight-through gradient. See Section 3.2.
        quantized_x = x + (quantized_x - x).detach()

        if self.use_ema and self.training:
            with torch.no_grad():
                # See Appendix A.1 of "Neural Discrete Representation Learning".

                # Cluster counts.
                encoding_one_hots = F.one_hot(
                    encoding_indices, self.num_embeddings
                ).type(flat_x.dtype)
                n_i_ts = encoding_one_hots.sum(0)
                # Updated exponential moving average of the cluster counts.
                # See Equation (6).
                self.N_i_ts(n_i_ts)

                # Exponential moving average of the embeddings. See Equation (7).
                embed_sums = flat_x.transpose(0, 1) @ encoding_one_hots
                self.m_i_ts(embed_sums)

                # This is kind of weird.
                # Compare: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L270
                # and Equation (8).
                N_i_ts_sum = self.N_i_ts.average.sum()
                N_i_ts_stable = (
                    (self.N_i_ts.average + self.epsilon)
                    / (N_i_ts_sum + self.num_embeddings * self.epsilon)
                    * N_i_ts_sum
                )
                self.e_i_ts = self.m_i_ts.average / N_i_ts_stable.unsqueeze(0)
        
        return (
            quantized_x,
            dictionary_loss,
            commitment_loss,
            encoding_indices,
        )


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_downsampling_layers,
        num_residual_layers,
        num_residual_hiddens,
        embedding_dim,
        num_embeddings,
        use_ema,
        decay,
        epsilon,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels,
            num_hiddens,
            num_downsampling_layers,
            num_residual_layers,
            num_residual_hiddens,
        )
        self.pre_vq_conv = nn.Conv1d(
            in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1
        )
        self.vq = VectorQuantizer(
            embedding_dim, num_embeddings, use_ema, decay, epsilon
        )
        self.decoder = Decoder(
            embedding_dim,
            num_hiddens,
            num_downsampling_layers,
            num_residual_layers,
            num_residual_hiddens,
        )

    def quantize(self, x):
        z = self.pre_vq_conv(self.encoder(x))
        (z_quantized, dictionary_loss, commitment_loss, encoding_indices) = self.vq(z)
        return (z_quantized, dictionary_loss, commitment_loss, encoding_indices)

    def forward(self, x):
        (z_quantized, dictionary_loss, commitment_loss, _) = self.quantize(x)
        x_recon = self.decoder(z_quantized)
        return {
            "dictionary_loss": dictionary_loss,
            "commitment_loss": commitment_loss,
            "x_recon": x_recon,
        }