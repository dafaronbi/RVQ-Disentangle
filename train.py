import dataset
import model
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import yaml
import auraloss
import dac
import librosa
from audiotools import AudioSignal
import numpy as np
import matplotlib.pyplot as plt

def grab_buffer(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def make_pyin_img(samples, pyin):
    f0 = pyin[0].cpu().detach().numpy()
    times = librosa.times_like(f0)

    y = samples.cpu().numpy()


    D = librosa.amplitude_to_db(np.abs(librosa.stft(y.T)), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
    ax.set(title='pYIN fundamental frequency estimation')
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


parser = argparse.ArgumentParser(description='Load training parameters yml')
parser.add_argument('-p', '--params', help="parameter yml file for training model", default="training_parameters/default.yml")
args = parser.parse_args()

#load parameters for training model
with open(args.params) as f:
    training_params = yaml.safe_load(f)

# How many GPUs are there?
print("GPU COUNT: " + str(torch.cuda.device_count()))

# #set training parameters
# epochs = int(training_params["epochs"])
# lr = float(training_params["lr"])
save_path = training_params["save_path"]
data_path = training_params["data_path"]
v_data_path = training_params["validation_data_path"]
batch_size = training_params["batch_size"]

#log for tensorboard
writer = SummaryWriter(training_params["tb_path"])


#get training and validation datasets
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data = dataset.NSynth(data_path)
train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

#data setup
v_data = dataset.NSynth(v_data_path)
valid_loader = torch.utils.data.DataLoader(v_data, batch_size=1, shuffle=True)

#create style encoders
timbre_enc = model.style_enc(20).to(device)
pitch_enc = model.style_enc(1).to(device)
loudness_enc = model.style_enc(1).to(device)

#create content enc
content_enc = model.content_enc().to(device)

#create decoder
decoder = model.decoder(1046).to(device)

#create DAC encoder and decoder
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path).to(device)
model.eval()

# Initialize optimizer.
lr = training_params["lr"]
optimizer = optim.Adam(list(timbre_enc.parameters()) + list(pitch_enc.parameters()) + list(loudness_enc.parameters()) 
+ list(content_enc.parameters()) + list(decoder.parameters()), lr=lr)

criterion = torch.nn.MSELoss()

# Train model.
eval_every = 1
best_train_loss = float("inf")
timbre_enc.train()
pitch_enc.train()
loudness_enc.train()
content_enc.eval()
decoder.eval()

for epoch in range(training_params["epochs"]):

    print(f"Epoch: {epoch}")

    total_train_loss = 0
    total_recon_loss = 0
    total_mfcc_loss = 0
    total_pitch_loss = 0
    total_loudness_loss = 0
    total_commitment_loss = 0
    total_codebook_loss = 0

    for (batch_idx, train_tensors) in enumerate(train_loader):
        loss = 0

        samples, mfcc, pitch, rms = train_tensors
        
        samples = samples.to(device)
        mfcc = mfcc.to(device)
        pitch = pitch.to(device).to(torch.float)
        rms = rms.to(device)

        z, codes, latents, _, _ = model.encode(samples[:,None,:])

        t_emb = timbre_enc(z)
        p_emb = pitch_enc(z)
        l_emb = loudness_enc(z)
        c_emb, _, vq_losses = content_enc(z)


        commitment_loss = vq_losses["commitment"]
        codebook_loss = vq_losses["codebook"]

        mfcc_loss = criterion(mfcc[..., :t_emb.shape[-1]], t_emb)
        pitch_loss = criterion(pitch[..., :p_emb.shape[-1]], p_emb)
        loudness_loss = criterion(rms[..., :l_emb.shape[-1]], l_emb)

        emb = torch.cat((t_emb, p_emb, l_emb, c_emb), 1)

        z_rec = decoder(emb)
        recon_loss = criterion(z, z_rec)

        if epoch > 100:
            content_enc.train()
            decoder.train()
            loss = recon_loss + mfcc_loss + pitch_loss + loudness_loss + commitment_loss + codebook_loss
            loss.sum().backward()
            optimizer.step()
        
        else:
            loss = mfcc_loss + pitch_loss + loudness_loss
            loss.sum().backward()
            optimizer.step()

        total_train_loss += loss.sum().item()
        total_recon_loss += recon_loss.item()
        total_mfcc_loss += mfcc_loss.item()
        total_pitch_loss += pitch_loss.item()
        total_loudness_loss += loudness_loss.item()
        total_commitment_loss += commitment_loss.sum().item()
        total_codebook_loss += codebook_loss.sum().item()

    writer.add_scalar("Loss/Total", total_train_loss, epoch)
    writer.add_scalar("Loss/recon", total_recon_loss, epoch)
    writer.add_scalar("Loss/mfcc", total_mfcc_loss, epoch)
    writer.add_scalar("Loss/pitch", total_pitch_loss, epoch)
    writer.add_scalar("Loss/loudness", total_loudness_loss, epoch)
    writer.add_scalar("Loss/commitmentl", total_commitment_loss, epoch)
    writer.add_scalar("Loss/codebook", total_codebook_loss, epoch)


    if  epoch % training_params["save_frequency"] == 0:
        if total_train_loss < best_train_loss:

            best_train_loss = total_train_loss
            torch.save(timbre_enc, training_params["save_path"] + "_t_enc.pt")
            torch.save(pitch_enc, training_params["save_path"] + "_p_enc.pt")
            torch.save(loudness_enc, training_params["save_path"] + "_l_enc.pt")
            torch.save(content_enc, training_params["save_path"] + "_c_enc.pt")
            torch.save(decoder, training_params["save_path"] + "_dec.pt")

            for i in range(3):
                samples,mfcc,pyin, rms = next(iter(valid_loader))

                samples = samples.to(device)
                mfcc = mfcc.to(device)
                pyin = pyin.to(device)
                rms = rms.to(device)

                z, codes, latents, _, _ = model.encode(samples[:,None,:])

                c_emb, _, vq_losses = content_enc(z)
                t_emb = timbre_enc(z)
                p_emb = pitch_enc(z)
                l_emb = loudness_enc(z)

                emb = torch.cat((t_emb, p_emb, l_emb, c_emb), 1)
                z_rec = decoder(emb)

                output_audio = model.decode(z_rec)

                writer.add_audio(f"Audio/Input-{i}:"  , samples)
                writer.add_audio(f"Audio/Reconstruction-{i}" , output_audio[0])

                samples = samples[0]
                pyin =  (pyin * (data.pitch_max - data.pitch_min) - data.pitch_min)[0]
                p_pred = (p_emb * (data.pitch_max - data.pitch_min) - data.pitch_min)[0]
                mfcc =  mfcc * (data.mfcc_max - data.mfcc_min) - data.mfcc_min
                t_pred = (t_emb * (data.pitch_max - data.pitch_min) - data.pitch_min)


                writer.add_image(f"Pitch/Input-{i}", make_pyin_img(samples, pyin), dataformats='HWC')
                writer.add_image(f"Pitch/Reconstruction-{i}", make_pyin_img(samples, p_pred), dataformats='HWC')

                writer.add_image(f"MFCC/Input-{i}", make_mfcc_img(mfcc), dataformats='HWC')
                writer.add_image(f"MFCC/Reconstruction-{i}", make_mfcc_img(t_pred), dataformats='HWC')

                writer.add_image(f"RMS/Input-{i}", make_rms_img(rms), dataformats='HWC')
                writer.add_image(f"RMS/Reconstruction-{i}", make_rms_img(l_emb), dataformats='HWC')
    

# Generate and save reconstructions.
# network.eval()

# audio = next(iter(train_loader))
# out = network(audio[0].to(device))

# writer.add_audio("Input Audio "  , audio[0][0], 0, 16000)
# writer.add_audio("Reconstuction " , out["x_recon"][0], 0, 16000)

writer.flush()
writer.close()