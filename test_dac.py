import dac
from audiotools import AudioSignal

import d_models
import torch

#create style encoders
timbre_enc = d_models.style_enc(128)
pitch_enc = d_models.style_enc(1)
loudness_enc = d_models.style_enc(1)

#create content enc
content_enc = d_models.content_enc()

#create decoder
decoder = d_models.decoder(1154)

#make test embedding
test_e = torch.zeros(5, 1024,2411)

#get encoder outputs
out_ts = timbre_enc(test_e)
out_ps  = pitch_enc(test_e)
out_ls = loudness_enc(test_e)

out_c = content_enc(test_e)[0]

#decode
print(decoder(torch.cat((out_ts, out_ps, out_ls, out_c),1)).shape)


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
