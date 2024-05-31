import librosa
import numpy as np
import glob
import os
import dac
import argparse
import torch

parser = argparse.ArgumentParser(description='Data Directory')
parser.add_argument('-d', '--data_dir', help="directory of data to save features", default="/vast/df2322/data/Nsynth/nsynth-train")
args = parser.parse_args()

data_dir = args.data_dir

#get device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#create DAC encoder and decoder
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path).to(device)
model.eval()

filenames = glob.glob(os.path.join(data_dir, "audio/*.wav"))

for filename in filenames:
    print(filename[:-4])
    samples, sr = librosa.load(filename, sr=44100)
    frame_size = len(samples)
    t_samples = torch.tensor(samples).to(device)
    z, codes, latents, _, _ = model.encode(t_samples[None,None,:])

    # mfcc = librosa.feature.mfcc(y=samples, sr = sr)
    # pyin,_,_ = librosa.pyin(y=samples, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr = sr, fill_na=0.0)
    # pyin = np.expand_dims(pyin,0)
    # rms = librosa.feature.rms(y=samples)

    np.save(filename[:-4] + "_z", z.detach().cpu().numpy())
    # np.save(filename[:-4] + "_mfcc", mfcc)
    # np.save(filename[:-4] + "_pitch", pyin)
    # np.save(filename[:-4] + "_rms", rms)