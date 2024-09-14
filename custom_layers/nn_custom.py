# Copyright 2020 InterDigital R&D and Télécom Paris.
# Author: Ondřej Cífka
# License: Apache 2.0

import torch
from torch import nn


class RNNWrapper(nn.Module):
    """Wrapper for plugging an RNN into a CNN."""

    def __init__(self, rnn=None, return_state=False, return_output=None):
        super().__init__()
        if rnn is not None:
            self.rnn = rnn
        else:
            raise RuntimeError(f'Input to RNN should not be NONE')
        self.return_state = return_state
        self.return_output = return_output or not return_state

    def forward(self, input):
        output, state = self.rnn(input.transpose(1, 2))
        output = output.transpose(1, 2)
        state = state.transpose(0, 1).reshape(input.shape[0], -1)
        if self.return_output and self.return_state:
            return output, state
        if self.return_state:
            return state
        return output


class ResidualWrapper(nn.Module):
    """Wrapper for adding a skip connection around a module."""

    def __init__(self, module=None):
        super().__init__()
        if module is not None:
            self.module = module
        # elif 'module' in self._cfg:
        #     self.module = self._cfg['module'].configure()
        # elif 'modules' in self._cfg:
        #     self.module = nn.Sequential(*self._cfg['modules'].configure_list())

    def forward(self, input):
        output = self.module(input)
        if output.shape != input.shape:
            raise RuntimeError(f'Expected output to have shape {input.shape}, got {output.shape}')
        return output + input

class SwapAxes(nn.Module):
    def __init__(self, ax):
        super().__init__()
        self.ax = ax

    def forward(self, x):
        return x.swapaxes(*self.ax)

#Gru that doesn't return hidden so it can be used in sequential
class GRUWrap(nn.Module):
    """Wrapper for adding a skip connection around a module."""

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False):
        super().__init__()
        
        self.gruWrap = nn.GRU(input_size,hidden_size,num_layers,batch_first)

    def forward(self, input):
        out,_ = self.gruWrap(input.transpose(1, 2))
        return out.transpose(1,2)