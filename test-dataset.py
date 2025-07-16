import torch
import dataset
import dac
import librosa
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from audiotools import AudioSignal
from sklearn.manifold import TSNE
import model
import matplotlib.ticker as ticker
import soundfile as sf


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

#get training and validation datasets
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
gpu_count = torch.cuda.device_count()

dac_model_path = dac.utils.download(model_type="44khz")
dac_model = dac.DAC.load(dac_model_path).to(device)
dac_model.eval()


# print([f"train_tensor_JC_{idx}.pt" for idx in range(24)])
# exit()
data = dataset.NSynth_analysis([f"data/train_tensor_JC_2.pt" for idx in range(24)], instruments=None) #[697])
loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, drop_last=True, num_workers=0*gpu_count)


num_to_instrument = {0: "bass", 1 : "brass", 2 : "flute", 3 : "guitar", 4 : "keyboard", 5 : "mallet", 6 : "organ", 7 : "reed", 
    8 : "string", 9 : "synth_lead", 10 : "vocal"}
pitches = {}
for l in loader:
    # print(f' {l["instrument"].item()} - {l["pitch"].item()} - {l["velocity"].item()}')
    # print(l["instrument_family_str"][0][0])
    # print(l)
    # print(l)
    # print(l["instrument"])
    # print(l["instrument_family_str"][0][0])
    # if l["instrument_family_str"][0][0] == "string" and l["instrument_source_str"][0][0] == "acoustic":  
    if l["instrument"][0][0].item() in pitches:
        pitches[l["instrument"][0][0].item()].append(l["pitch"][0][0].item())
    else:
        pitches[l["instrument"][0][0].item()] = [l["pitch"][0][0].item()]

x = []
y = []

for key in pitches:
    for p in pitches[key]:
        x.append(p)
        y.append(key)

max_pitches = []
for key in pitches:
    max_pitches.append(np.max(pitches[key]))
    print(np.max(pitches[key]))

print(f"MIN OF MAX PITCHES IS: {min(max_pitches)}")
exit()
print(sorted(x))
# d =  np.array([ [(p,key) for p in pitches[key]] for key in pitches]).reshape(-1,2)
# print(d)
# x = d[:,0]
# y = d[:,1]

# fig, ax = plt.subplots()
# caxes = ax.hist2d(x, y, bins=(len(range(21,109)), 11), cmap=plt.cm.Blues_r, density=True)
# # cbar = fig.colorbar(caxes)
# # cbar.ax.set_ylabel('Probability', rotation=270)
# ax.set(title=f'Distribution of Pitches instrument qualities')
# ax.set_yticks(np.arange(11))
# ax.set_yticklabels(num_to_instrument.values())
# ax.set_xlabel("Pitch (Midi Value)")
# ax.set_ylabel("Instrument Family")
# plt.savefig(f"pitch_histogram 2d vs instrument family.png")

for key in pitches:
    fig, ax = plt.subplots()

    ax.hist(pitches[key], bins=list(range(21,120)), density=True)
    ax.set(title=f'Distribution of String Acoustic Subtype')

    plt.savefig(f"pitch_histogram String Acoustic.png")

# for key in pitches:
#     fig, ax = plt.subplots()

#     ax.hist(pitches[key], bins=list(range(21,109)), density=True)
#     ax.set(title=f'Distribution of Pitches instrument qualities {key}')

#     plt.savefig(f"pitch_histogram i_qualities {key}.png")


