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
        return self.encode_content(input_e), 0, {"commitment": torch.tensor(0), "codebook": torch.tensor(0)} #encoded_c, _, losses_c

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