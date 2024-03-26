import torch
import dataset
import dac
import librosa
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

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
    

sr = 44100

#data setup
data_path = "/vast/df2322/data/Nsynth/nsynth-valid"
data = dataset.NSynth(data_path)
train_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

#set device used to perform training
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#log for tensorboard
writer = SummaryWriter("tensorboard/inference_runs")

#load saved models
c_enc = torch.load("saved_models/c_enc.pt").to(device)
t_enc = torch.load("saved_models/t_enc.pt").to(device)
p_enc = torch.load("saved_models/p_enc.pt").to(device)
l_enc = torch.load("saved_models/l_enc.pt").to(device)
dec = torch.load("saved_models/dec.pt").to(device)

#create DAC encoder and decoder
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path).to(device)
model.eval()

for i in range(3):
    samples,mfcc,pyin, rms = next(iter(train_loader))

    samples = samples.to(device)
    mfcc = mfcc.to(device)
    pyin = pyin.to(device)
    rms = rms.to(device)

    z, codes, latents, _, _ = model.encode(samples[:,None,:])

    c_emb, _, vq_losses = c_enc(z)
    t_emb = t_enc(z)
    p_emb = p_enc(z)
    l_emb = l_enc(z)

    emb = torch.cat((t_emb, p_emb, l_emb, c_emb), 1)
    z_rec = dec(emb)

    output_audio = model.decode(z)

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

writer.flush()
writer.close()