import argparse
import logging
import os
import random

import confugue
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torch.utils.tensorboard
from torch import nn

@confugue.configurable
class style_enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.style_encoder_1d = nn.Sequential(*self._cfg['style_encoder_1d'].configure_list())
        self.style_encoder_rnn = self._cfg['style_encoder_rnn'].maybe_configure(nn.GRU, batch_first=True)
        self.style_encoder_0d = nn.Sequential(*self._cfg['style_encoder_0d'].configure_list())

    def encode_style(self, input, length):
        encoded = self.style_encoder_1d(input)

        # Mask positions corresponding to padding
        length = (length // (input.shape[2] / encoded.shape[2])).to(torch.int)
        mask = (torch.arange(encoded.shape[2], device=encoded.device) < length[:, None])[:, None, :]
        encoded *= mask

        if self.style_encoder_rnn is not None:
            encoded = encoded.transpose(1, 2)
            encoded = nn.utils.rnn.pack_padded_sequence(
                encoded, length.clamp(min=1),
                batch_first=True, enforce_sorted=False)
            _, encoded = self.style_encoder_rnn(encoded)
            # Get rid of layer dimension
            encoded = encoded.transpose(0, 1).reshape(input.shape[0], -1)
        else:
            # Compute the Gram matrix, normalized by the length squared
            encoded /= mask.sum(dim=2, keepdim=True) + torch.finfo(encoded.dtype).eps
            encoded = torch.matmul(encoded, encoded.transpose(1, 2))
        encoded = encoded.reshape(encoded.shape[0], -1)

        encoded = self.style_encoder_0d(encoded)

        return encoded, {}

    def foward(input_e):
        encoded_s, losses_s = self.encode_style(input_s, length_s)
        return
        
@confugue.configurable
class content_enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.content_encoder = nn.Sequential(*self._cfg['content_encoder'].configure_list())
        self.vq = self._cfg['vq'].configure(VQEmbedding, axis=1)

    def encode_content(self, input):
        encoded = self.content_encoder(input)
        if self.vq is None:
            return encoded, encoded, {}
        return self.vq(encoded)
    
    def foward(input_e):
        encoded_c, _, losses_c = self.encode_content(input_c)
        return

@confugue.configurable
class decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_modules = nn.ModuleList([
            nn.Sequential(*self._cfg['decoder'][i].configure_list())
            for i in range(len(self._cfg['decoder']))
        ])
    
    def decode(self, encoded_c, encoded_s, length=None, max_length=None):
        encoded_s = encoded_s[:, :, None]

        decoded = encoded_c
        for module in self.decoder_modules:
            decoded = torch.cat([
                decoded,
                encoded_s.expand(-1, -1, decoded.shape[-1])
            ], axis=1)
            decoded = module(decoded)

    def foward(input_e):
        decoded = self.decode(encoded_c, encoded_s, length=length_c, max_length=input_c.shape[2])
        return decoded

# @confugue.configurable
# class Model(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.content_encoder = nn.Sequential(*self._cfg['content_encoder'].configure_list())
#         self.vq = self._cfg['vq'].configure(VQEmbedding, axis=1)

#         self.style_encoder_1d = nn.Sequential(*self._cfg['style_encoder_1d'].configure_list())
#         self.style_encoder_rnn = self._cfg['style_encoder_rnn'].maybe_configure(nn.GRU,
#                                                                                 batch_first=True)
#         self.style_encoder_0d = nn.Sequential(*self._cfg['style_encoder_0d'].configure_list())

#         self.decoder_modules = nn.ModuleList([
#             nn.Sequential(*self._cfg['decoder'][i].configure_list())
#             for i in range(len(self._cfg['decoder']))
#         ])

#     def forward(self, input_c, input_s, length_c, length_s, return_losses=False):
#         encoded_c, _, losses_c = self.encode_content(input_c)
#         encoded_s, losses_s = self.encode_style(input_s, length_s)
#         decoded = self.decode(encoded_c, encoded_s, length=length_c, max_length=input_c.shape[2])

#         if not return_losses:
#             return decoded

#         losses = {
#             'reconstruction': ((decoded - input_c) ** 2).mean(axis=1),
#             **losses_c
#         }

#         # Sum losses over time and batch, normalize by total time
#         assert all(len(loss.shape) == 2 for loss in losses.values())
#         losses = {name: loss.sum() / (length_c.sum() + torch.finfo(loss.dtype).eps)
#                   for name, loss in losses.items()}

#         # Add losses which don't have the time dimension
#         assert all(len(loss.shape) == 1 for loss in losses_s.values())
#         losses.update({name: loss.mean() for name, loss in losses_s.items()})

#         return decoded, losses

#     def encode_content(self, input):
#         encoded = self.content_encoder(input)
#         if self.vq is None:
#             return encoded, encoded, {}
#         return self.vq(encoded)

#     def encode_style(self, input, length):
#         encoded = self.style_encoder_1d(input)

#         # Mask positions corresponding to padding
#         length = (length // (input.shape[2] / encoded.shape[2])).to(torch.int)
#         mask = (torch.arange(encoded.shape[2], device=encoded.device) < length[:, None])[:, None, :]
#         encoded *= mask

#         if self.style_encoder_rnn is not None:
#             encoded = encoded.transpose(1, 2)
#             encoded = nn.utils.rnn.pack_padded_sequence(
#                 encoded, length.clamp(min=1),
#                 batch_first=True, enforce_sorted=False)
#             _, encoded = self.style_encoder_rnn(encoded)
#             # Get rid of layer dimension
#             encoded = encoded.transpose(0, 1).reshape(input.shape[0], -1)
#         else:
#             # Compute the Gram matrix, normalized by the length squared
#             encoded /= mask.sum(dim=2, keepdim=True) + torch.finfo(encoded.dtype).eps
#             encoded = torch.matmul(encoded, encoded.transpose(1, 2))
#         encoded = encoded.reshape(encoded.shape[0], -1)

#         encoded = self.style_encoder_0d(encoded)

#         return encoded, {}

#     def decode(self, encoded_c, encoded_s, length=None, max_length=None):
#         encoded_s = encoded_s[:, :, None]

#         decoded = encoded_c
#         for module in self.decoder_modules:
#             decoded = torch.cat([
#                 decoded,
#                 encoded_s.expand(-1, -1, decoded.shape[-1])
#             ], axis=1)
#             decoded = module(decoded)

#         # Make sure the output tensor has the same shape as the input tensor
#         if max_length is not None or length is not None:
#             if max_length is None:
#                 max_length = length.max()

#             decoded = decoded.narrow(-1, 0, max_length)

#         # Make sure output lengths are the same as input lengths
#         if length is not None:
#             mask = (torch.arange(max_length, device=decoded.device) < length[:, None])[:, None, :]
#             decoded *= mask

#         return decoded