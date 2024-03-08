import dac
from audiotools import AudioSignal

import d_models
import torch
import confugue

cfg_path = 'config.yaml'
cfg = confugue.Configuration.from_yaml_file(cfg_path)
network = cfg.configure(d_models.style_enc, config_path=cfg_path, logdir='logs')


exit()

# Download a model
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path)

model.to('cuda')

# Load audio signal file
signal = AudioSignal('audio/survival.mp3')

signal = signal.resample(44100).to_mono().truncate_samples(44100)

# Encode audio signal as one long file
# (may run out of GPU memory on long files)
signal.to(model.device)
print(signal.audio_data)
print(signal.sample_rate)
print(signal.audio_data.shape)
x = model.preprocess(signal.audio_data, signal.sample_rate)
z, codes, latents, _, _ = model.encode(x)
print(z.shape)
print(codes.shape)
print(latents.shape)
