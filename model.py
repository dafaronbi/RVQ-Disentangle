# Ported from: https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb.

import torch

from torch import nn
from torch.nn import functional as F

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

        self.emb_dim = 1024

        model_path = dac.utils.download(model_type="44khz") 
        self.dacModel = dac.DAC.load(model_path)
        self.dacModel.eval()

        self.pitch_predict = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn_custom.ResidualWrapper(
            nn.Sequential(
            nn.SyncBatchNorm(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=3, dilation=3),
        )),
        nn.Conv1d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=6, dilation=6),
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
            nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=5, stride=1, padding=2, dilation=1),
            nn_custom.ResidualWrapper(
            nn.Sequential(
            nn.SyncBatchNorm(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=3, dilation=3),
        )),
        nn.Conv1d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=6, dilation=6),
        nn.SyncBatchNorm(32),
        nn.ReLU(),
        nn.Conv1d(in_channels=32, out_channels=self.emb_dim, kernel_size=3, stride=1, padding=9, dilation=9),
        nn.ReLU()
            )

        # self.rest_transform =nn.Conv1d(in_channels=self.emb_dim+1, out_channels=self.emb_dim, kernel_size=5, stride=1, padding=2, dilation=1)

        self.decoder = nn.Sequential(
                nn.SyncBatchNorm(self.emb_dim),
                nn.ConvTranspose1d(self.emb_dim, 256, 3, 1, 1, dilation=1),
                nn_custom.ResidualWrapper(
                    nn.Sequential(
                    nn.SyncBatchNorm(256),
                    nn.ReLU(),
                    nn.ConvTranspose1d(256, 256, 7, 1, 9, dilation=3),
                    nn.ReLU(),
                )
                ),
                nn_custom.ResidualWrapper(
                    nn.Sequential(
                    nn.SyncBatchNorm(256),
                    nn.ReLU(),
                    nn.ConvTranspose1d(256, 256,7, 1, 18, dilation=6),
                    nn.ReLU(),
                ),
                ),
                # nn_custom.ResidualWrapper(
                    nn.Sequential(
                    nn.SyncBatchNorm(256),
                    nn.ReLU(),
                    nn.ConvTranspose1d(256, 512,7, 1, 27, dilation=9),
                    nn.ReLU(),
                ),
                # ),
                # nn_custom.ResidualWrapper(
                    nn.Sequential(
                    nn.SyncBatchNorm(512),
                    nn.ReLU(),
                    nn.ConvTranspose1d(512, 1024,7, 1, 36, dilation=12),
                    nn.ReLU(),
                ),
                # ),
                nn_custom.GRUWrap(1024,1024,1, batch_first=True),
                nn.ReLU(),
                nn.ConvTranspose1d(1024, 1024*9, kernel_size=7, stride=1, padding=45, dilation=15),
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
        z = self.dacModel.quantizer.from_codes(z)[0]

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

        #predict pitch
        p_predict = self.pitch_predict(z)
        p_predict = torch.argmax(p_predict, dim=1).float().unsqueeze(1)

        #predict mfcc 
        m_predict = self.mfcc_predict(z)

        # #predict rms
        r_predict = self.rms_predict(z)


        if mfcc != None:
            mfcc = mfcc[..., :z.shape[-1]]
        
        if rms != None:
            rms = rms[..., :z.shape[-1]]

        if p != None:
            p = p.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
            # p[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)
            p_embedding = self.encode_pitch(p)
        else:
            p_embedding = self.encode_pitch(p_predict)

        # if mfcc != None:
        #     m_embedding = self.encode_mfcc(mfcc)
        # else:
        #     m_embedding = self.encode_mfcc(m_predict)
        #     mfcc = m_predict

        if rms != None:
            r_embedding = self.encode_rms(rms)
        else:
            r_embedding = self.encode_rms(r_predict)

        # if p != None:
        #     p_predict = p

        # if mfcc != None:
        #     m_predict = mfcc

        # if rms != None:
        #     r_predict = rms

        if inst != None:
            inst = inst.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
            m_embedding = self.encode_mfcc(inst)
        else:
            inst = torch.tensor([759]).to(self.device)
            inst = i_prime.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
            m_embedding = self.encode_mfcc(inst)

        #get the rest embedding form the rest encoder (should learn to subtract pitch)
        rest_embedding = self.encode_rest(z)
        p_start = p_start.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
        p_start_embedding = self.encode_pitch(p_start)
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
        with torch.no_grad():
            z = self.dacModel.quantizer.from_codes(z)[0]

        #loss categorical and regression loss functions
        C_loss = nn.NLLLoss()
        MSE_loss = nn.MSELoss()

        #predict pitch
        p_predict = self.pitch_predict(z)

        # #predict mfcc 
        m_predict = self.mfcc_predict(z)

        # #predict rms
        r_predict = self.rms_predict(z)

        p = p.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
        p[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)

        mfcc = mfcc[..., :z.shape[-1]]
        rms = rms[..., :z.shape[-1]]

        pitch_predict_loss = C_loss(p_predict, p[:,0,:].long())
        mfcc_predict_loss = MSE_loss(m_predict, mfcc)
        rms_predict_loss = MSE_loss(r_predict, rms)

        loss = {"p_predict": pitch_predict_loss,
                "m_predict": mfcc_predict_loss,
                "r_predict": rms_predict_loss,
                }

        predict = {"pitch" : p_predict,
                    "mfcc" : m_predict,
                    "rms" : r_predict
        }

        return loss, predict

    def train_recon(self, z,  z_prime, p_prime,mfcc_prime, rms_prime):
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


    def forward(self, z, p, mfcc, rms, inst, z_prime, p_prime,mfcc_prime, rms_prime, inst_prime):
        
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

        #predict pitch
        # p_predict = self.pitch_predict(z_prime)
        # p_predict = torch.argmax(p_predict, dim=1).float().unsqueeze(1)

        # #predict mfcc 
        # m_predict = self.mfcc_predict(z_prime)

        # #predict rms
        # r_predict = self.rms_predict(z_prime)

        p = p.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
        # p[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)
        p_prime = p_prime.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
        # p_prime[:,:, ((z.shape[-1]*self.pc_num)//self.pc_denom):] = torch.tensor(0).to(self.device)
        mfcc = mfcc[..., :z.shape[-1]]
        mfcc_prime = mfcc_prime[..., :z.shape[-1]]
        rms = rms[..., :z.shape[-1]]
        rms_prime = rms_prime[..., :z.shape[-1]]
        inst = inst.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
        inst_prime = inst_prime.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()

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
        r_embedding = self.encode_rms(rms)
        m_embedding = self.encode_mfcc(inst)
        p_embedding_prime = self.encode_pitch(p_prime)
        r_embedding_prime = self.encode_rms(rms_prime)
        m_embedding_prime = self.encode_mfcc(inst_prime)
        

        #get the rest embedding form the rest encoder (should learn to subtract pitch)

        rest_embedding = self.encode_rest(z)
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
        pitch_ce_loss = COS_loss(p_embedding.view(z.shape[0], p_embedding.shape[1]*p_embedding.shape[2]), rest_embedding.view(z.shape[0], rest_embedding.shape[1]*rest_embedding.shape[2]), torch.full((z.shape[0],), -1).to(self.device))
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
                "p_ce_loss": pitch_ce_loss,
                # "m_ce_loss": mfcc_ce_loss,
                # "r_ce_loss": rms_ce_loss,
                # "pm_ce_loss": pm_ce_loss,
                # "pr_ce_loss": pr_ce_loss,
                # "mr_ce_loss": mr_ce_loss,
                }

        predict = {"z" : z_hat,
                    "pitch" : p_prime, #p_predict,
                    "mfcc" : mfcc_prime, #m_predict,
                    "rms" : rms_prime, #r_predict
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