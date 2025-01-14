import dataset
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import argparse

import yaml
import auraloss
import dac
import librosa
from audiotools import AudioSignal
import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchmetrics.clustering import MutualInfoScore
import matplotlib.pyplot as plt
import random

import math
from typing import Iterator, Optional, TypeVar
import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
import datetime
import sys
import model
import pesq

_T_co = TypeVar("_T_co", covariant=True)

class NonRedundantSampler(DistributedSampler):
    def __init__(self, dataset: torch.utils.data.Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        # super().__init__(self, dataset, num_replicas, rank, shuffle,
        #                  seed, drop_last)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        # Initialize your stuff

    def __iter__(self) -> Iterator[_T_co]:

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        # assert len(indices) == self.num_samples
        ip_pair = set()
        indices_to_keep = list()
        for idx in indices:
            if (self.dataset[idx][1].item(), self.dataset[idx][4].item()) not in ip_pair:
                ip_pair.add((self.dataset[idx][1].item(), self.dataset[idx][4].item()))
                indices_to_keep.append(idx)
        
        #check for drop last again
        if self.drop_last and len(indices_to_keep) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            num_samples = math.ceil(
                (len(indices_to_keep) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            num_samples = math.ceil(len(indices_to_keep) / self.num_replicas)
        total_size = num_samples * self.num_replicas

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = total_size - len(indices_to_keep)
            if padding_size <= len(indices_to_keep):
                indices_to_keep += indices_to_keep[:padding_size]
            else:
                indices_to_keep += (indices_to_keep * math.ceil(padding_size / len(indices_to_keep)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices_to_keep = indices_to_keep[: total_size]
        assert len(indices_to_keep) == total_size

        return iter(indices_to_keep)


def ddp_setup(rank: int, world_size: int):
    """
    Args:
    rank: Unique identifier of each process
    world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def ddp_cleanup():
    destroy_process_group()

def grab_buffer(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def make_pitch_img(samples, p):
    f0 = p[0].cpu().detach().numpy()
    times = librosa.times_like(f0)

    y = samples.cpu().detach().numpy()


    D = librosa.amplitude_to_db(np.abs(librosa.stft(y.T)), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
    ax.set(title='fundamental frequency estimation')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    ax.legend(loc='upper right')

    fig.canvas.draw()

    return grab_buffer(fig)

def make_mfcc_img(mfcc):
    mfcc = mfcc[0].cpu().detach().numpy()

    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title='MFCC')

    fig.canvas.draw()

    return grab_buffer(fig)

def make_rms_img(rms):
    rms = rms[0].cpu().detach().numpy()
    
    fig, ax = plt.subplots()
    times = librosa.times_like(rms)
    ax.semilogy(times, rms[0], label='RMS Energy')
    ax.set(xticks=[])
    ax.legend()
    ax.label_outer()

    fig.canvas.draw()

    return grab_buffer(fig)

def calculate_pesq(pred, target, sr=441000):
    pred = librosa.resample(pred.detach().cpu().numpy(), orig_sr=sr, target_sr=16000)
    target = librosa.resample(target.detach().cpu().numpy(), orig_sr=sr, target_sr=16000)

    pesq_score = pesq.pesq(16000, pred, target, on_error=pesq.PesqError.RETURN_VALUES)
    
    #error scores = zero 
    if pesq_score < 0:
        pesq_score = 0

    return pesq_score


def main(rank, world_size):
    

    ddp_setup(rank, world_size)
    parser = argparse.ArgumentParser(description='Load training parameters yml')
    parser.add_argument('-p', '--params', help="parameter yml file for training model", default="training_parameters/default.yml")
    args = parser.parse_args()

    #load parameters for training model
    with open(args.params) as f:
        training_params = yaml.safe_load(f)

    save_path = training_params["save_path"]
    data_path = training_params["data_path"]
    v_data_path = training_params["validation_data_path"]
    batch_size = training_params["batch_size"]

    gpu_count = torch.cuda.device_count()
    # How many GPUs are there?
    if rank == 0:
        print("GPU COUNT: " + str(gpu_count))
        sys.stdout.flush()

    #get training and validation datasets
    device = rank
    if rank == 0:
        print("LOADING TRAINING DATA...")
        sys.stdout.flush()

    data = dataset.NSynth_transform_ram(training_params["data_path"], instruments=training_params["instruments"])
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=DistributedSampler(data), drop_last=False, num_workers=training_params["num_workers"])
    if rank == 0:
        print("DONE!!")
        sys.stdout.flush()

    if rank == 0:
        print("LOADING VALIDATION DATA...")
        sys.stdout.flush()
    v_data = dataset.NSynth_transform_ram(training_params["validation_data_path"], instruments=training_params["instruments"])
    valid_loader = torch.utils.data.DataLoader(v_data, batch_size=batch_size, sampler=DistributedSampler(v_data), drop_last=False, num_workers=training_params["num_workers"], shuffle=False)


    if rank == 0:
        print("DONE!!")
        sys.stdout.flush()

    v_frequency = training_params["v_frequency"]

    m = model.new_model(device).to(device)
    m = DDP(m, device_ids=[device], output_device=device, find_unused_parameters=True)

    #initialize dac model
    dac_model_path = dac.utils.download(model_type="44khz")
    dac_model = dac.DAC.load(dac_model_path).to(device)
    dac_model.eval()

    # Initialize optimizer.
    lr = training_params["lr"]

    optimizer = optim.Adam(m.parameters(), lr=lr)

    #log for tensorboard
    writer = SummaryWriter(training_params["tb_path"] + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    torch.cuda.empty_cache()

    if training_params["train_continue"]:
        m_load = torch.load(save_path).to(device)
        m.load_state_dict(m_load.state_dict())
        m = DDP(m, device_ids=[device], output_device=device, find_unused_parameters=True)
        optimizer = optim.Adam(m.parameters(), lr=lr)
        torch.distributed.barrier()

    epochs = training_params["epochs"]
    for epoch in range(1,epochs+1): 
        m.train()

        train_loader.sampler.set_epoch(epoch + training_params["encoder_epochs"])

        loss_total = 0
        loss_total_r = 0


        for (batch_idx, train_tensors) in enumerate(train_loader):
            z,p,mfcc,rms,inst,z_prime,p_prime,mfcc_prime,rms_prime,inst_prime = train_tensors   

            z = z.to(device)[:,0,:,:]
            p = p.to(device)
            z_prime = z_prime.to(device)[:,0,:,:]
            p_prime = p_prime.to(device)
            l,p = m(z,p,z_prime,p_prime)

            loss = l["t_predict"]


            loss.sum().backward()
            # torch.nn.utils.clip_grad_norm_(m.parameters(), 0.0001)


            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            

            loss_total += loss.item()

            loss_total_r += l["t_predict"].item()

            if training_params["verbos"]:
                if rank == 0:
                    print(f"sample {batch_idx + 1} out of {len(train_loader)}")
                    print(loss.item())
                    sys.stdout.flush()

        writer.add_scalar("Training/Total", loss_total / len(train_loader), epoch)
        writer.add_scalar("Training/NLL Loss", loss_total_r / len(train_loader), epoch)


        if (epoch % v_frequency) == 0:

            if rank == 0:
                torch.save(m.module, save_path)
            
            v_nll_total = 0
            v_accuracy_total = 0
            v_accuracy_0_total = 0
            v_accuracy_1_total = 0
            v_accuracy_2_total = 0
            v_accuracy_3_total = 0
            v_accuracy_4_total = 0
            v_accuracy_5_total = 0
            v_accuracy_6_total = 0
            v_accuracy_7_total = 0
            v_accuracy_8_total = 0
            v_pesq = 0

            #evaluate mode
            m.eval()

            for (batch_idx, train_tensors) in enumerate(valid_loader):
                z,p,mfcc,rms,inst,z_prime,p_prime,mfcc_prime,rms_prime,inst_prime = train_tensors   

                z = z.to(device)[:,0,:,:]
                p = p.to(device)
                z_prime = z_prime.to(device)[:,0,:,:]
                p_prime = p_prime.to(device)
                l,p = m(z,p,z_prime,p_prime)

                #calculate pesq of each audio sample
                with torch.no_grad():
                    for i,z_sample in enumerate(z_prime):
                        i_emb = dac_model.quantizer.from_codes(z_sample.unsqueeze(0))[0]
                        o_emb = dac_model.quantizer.from_codes(p["z"][i].unsqueeze(0))[0]
                        i_audio = dac_model.decode(i_emb)
                        o_audio = dac_model.decode(o_emb)
                        v_pesq += calculate_pesq(i_audio[0][0], o_audio[0][0])
                
                #normalize pesq score
                v_pesq /= len(z_prime)

                token_acc = (p['z'] == z_prime)

                #get negative log likelihood loss
                v_nll_total += l["t_predict"].item()

                #get total accuracy
                v_accuracy_total += ((torch.count_nonzero(token_acc) / torch.numel(z_prime))*100).item()

                #calculate accuracy per  token
                v_accuracy_0_total += ((torch.count_nonzero(token_acc[:, 0, :]) / torch.numel(z[:, 0, :]))*100).item()
                v_accuracy_1_total += ((torch.count_nonzero(token_acc[:, 1, :]) / torch.numel(z[:, 1, :]))*100).item()
                v_accuracy_2_total += ((torch.count_nonzero(token_acc[:, 2, :]) / torch.numel(z[:, 2, :]))*100).item()
                v_accuracy_3_total += ((torch.count_nonzero(token_acc[:, 3, :]) / torch.numel(z[:, 3, :]))*100).item()
                v_accuracy_4_total += ((torch.count_nonzero(token_acc[:, 4, :]) / torch.numel(z[:, 4, :]))*100).item()
                v_accuracy_5_total += ((torch.count_nonzero(token_acc[:, 5, :]) / torch.numel(z[:, 5, :]))*100).item()
                v_accuracy_6_total += ((torch.count_nonzero(token_acc[:, 6, :]) / torch.numel(z[:, 6, :]))*100).item()
                v_accuracy_7_total += ((torch.count_nonzero(token_acc[:, 7, :]) / torch.numel(z[:, 7, :]))*100).item()
                v_accuracy_8_total += ((torch.count_nonzero(token_acc[:, 8, :]) / torch.numel(z[:, 8, :]))*100).item()
            
            writer.add_scalar("Validation/NLL Loss", v_nll_total / len(valid_loader), epoch)
            writer.add_scalar("Validation/Accuracy", v_accuracy_total / len(valid_loader), epoch)
            writer.add_scalar("Validation/Accuracy Token 0", v_accuracy_0_total / len(valid_loader), epoch)
            writer.add_scalar("Validation/Accuracy Token 1", v_accuracy_1_total / len(valid_loader), epoch)
            writer.add_scalar("Validation/Accuracy Token 2", v_accuracy_2_total / len(valid_loader), epoch)
            writer.add_scalar("Validation/Accuracy Token 3", v_accuracy_3_total / len(valid_loader), epoch)
            writer.add_scalar("Validation/Accuracy Token 4", v_accuracy_4_total / len(valid_loader), epoch)
            writer.add_scalar("Validation/Accuracy Token 5", v_accuracy_5_total / len(valid_loader), epoch)
            writer.add_scalar("Validation/Accuracy Token 6", v_accuracy_6_total / len(valid_loader), epoch)
            writer.add_scalar("Validation/Accuracy Token 7", v_accuracy_7_total / len(valid_loader), epoch)
            writer.add_scalar("Validation/Accuracy Token 8", v_accuracy_8_total / len(valid_loader), epoch)
            writer.add_scalar("Validation/PESQ", v_pesq / len(valid_loader), epoch)

   
    ddp_cleanup()

    writer.flush()
    writer.close()





if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)