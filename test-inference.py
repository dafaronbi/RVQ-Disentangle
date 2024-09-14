import torch
import dataset
import dac
import librosa
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from audiotools import AudioSignal
from sklearn.manifold import TSNE


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

disentangle = torch.load("saved_models/disentangle_test.pt",map_location=device).to(device)
disentangle.device = device
disentangle.eval()

# print(disentangle)
# exit()

#log for tensorboard
writer = SummaryWriter("tensorboard/inference_runs")

data = dataset.NSynth_transform_ram(["test_tensor_JC_0.pt"], instruments=[759, 417, 644, 97])
test_loader = torch.utils.data.DataLoader(data, batch_size=5, shuffle=True, drop_last=True, num_workers=0*gpu_count)

#create DAC encoder and decoder
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path).to(device)
model.eval()

z,p,mfcc,rms,inst,z_prime,p_prime,mfcc_prime,rms_prime,inst_prime = next(iter(test_loader))

z = z.to(device)[:,0,:,:]
p = p.to(device)
mfcc = mfcc.to(device)
rms = rms.to(device)
inst = inst.to(device)
z_prime = z_prime.to(device)[:,0,:,:]
p_prime = p_prime.to(device)
mfcc_prime = mfcc_prime.to(device)
rms_prime = rms_prime.to(device)
inst_prime = inst_prime.to(device)
_,predict = disentangle(z,p,mfcc,rms,inst,z_prime, p_prime, mfcc_prime, rms_prime, inst_prime)

p_old = p
#make p span across time
p = p.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
p[:,:, ((z.shape[-1]*disentangle.pc_num)//disentangle.pc_denom):] = torch.tensor(0).to(disentangle.device)
p_prime = p_prime.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
p_prime[:,:, ((z.shape[-1]*disentangle.pc_num)//disentangle.pc_denom):] = torch.tensor(0).to(disentangle.device)



#turn p from midi to hertz
p = 440 * 2**((p-69)/12)
p_prime = 440 * 2**((p_prime-69)/12)
p_hat = 440 * 2**((predict["pitch"]-69)/12)

z_prime_codes = z_prime
z_prime = model.quantizer.from_codes(z_prime_codes[0].unsqueeze(0))[0]

z_codes = z
z = model.quantizer.from_codes(z_codes[0].unsqueeze(0))[0]


out_codes = predict["z"]
out = model.quantizer.from_codes(out_codes[0].unsqueeze(0))[0]

with torch.no_grad():
    input_audio = model.decode(z_prime)
    output_audio = model.decode(out)

writer.add_audio(f"Audio/Ground Truth:"  , input_audio[0])
writer.add_audio(f"Audio/Reconstruction" , output_audio[0])

samples = input_audio[0][0]
mfcc =  mfcc_prime * (data.mfcc_max - data.mfcc_min) + data.mfcc_min
mfcc_hat = (predict["mfcc"] * (data.mfcc_max - data.mfcc_min) + data.mfcc_min)


writer.add_image(f"Pitch/Ground Truth", make_pitch_img(samples, p_prime[0]), dataformats='HWC')
writer.add_image(f"Pitch/Reconstruction", make_pitch_img(samples, p_hat[0]), dataformats='HWC')

writer.add_image(f"MFCC/Ground Truth", make_mfcc_img(mfcc), dataformats='HWC')
writer.add_image(f"MFCC/Reconstruction", make_mfcc_img(mfcc_hat), dataformats='HWC')

writer.add_image(f"RMS/Ground Truth", make_rms_img(rms_prime), dataformats='HWC')
writer.add_image(f"RMS/Reconstruction", make_rms_img(predict["rms"]), dataformats='HWC')

z_hat = predict["z"]

#convert from discrete code to continuous embedding
# z_codes = z
# z = model.quantizer.from_codes(z)[0]


input_audio = model.decode(z[0].unsqueeze(0))
writer.add_audio(f"Audio/Input:"  , input_audio[0])

pitches = torch.tensor([num for num in [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,45]]).to(device)

for i in range(12):
    z_predict = disentangle.get_new_sample(z_codes[0].unsqueeze(0), pitches[i].unsqueeze(0), p_start=p_old[0].unsqueeze(0), mfcc=mfcc[0].unsqueeze(0), rms=rms[0].unsqueeze(0), inst=inst[0].unsqueeze(0))
    z_predict = model.quantizer.from_codes(z_predict)[0]
    out = model.decode(z_predict)

    # writer.add_audio(f"Audio/Output-{i}:"  , out[0])
    writer.add_audio(f"Audio/Predict: pitch = {pitches[i]}" , out[0])

#get all data
all_data = torch.utils.data.DataLoader(data, batch_size=len(data))
z,p,mfcc,rms,inst,z_prime,p_prime,mfcc_prime,rms_prime,inst_prime = next(iter(all_data))
z = z.to(device)[:,0,:,:]
p = p.to(device)
mfcc = mfcc.to(device)
rms = rms.to(device)
inst = inst.to(device)
z_prime = z_prime.to(device)[:,0,:,:]
p_prime = p_prime.to(device)
mfcc_prime = mfcc_prime.to(device)
rms_prime = rms_prime.to(device)
inst_prime = inst_prime.to(device)
disentangle(z,p,mfcc,rms,inst,z_prime, p_prime, mfcc_prime, rms_prime, inst_prime)

rest_emb = disentangle.rest_emb[:,:,100]
d_redux = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=40, n_iter=5000).fit_transform(rest_emb.cpu().detach().numpy())

fig, ax = plt.subplots()
cdict = {759: 'red', 417: 'blue', 644: 'green', 97: 'yellow'}
i_label = {759: 'bass_electronic_018', 417: 'bass_synthetic_033', 644: 'mallet_acoustic_062', 97: 'string_acoustic_012'}

for i in np.unique(inst.cpu().detach().numpy()):
    ix = np.where(inst.cpu().detach().numpy() == i)
    ax.scatter(d_redux.T[0][ix], d_redux.T[1][ix], label=i_label[i], color=cdict[i])

ax.set(title='Rest Embedding TSNE scatter plot')
ax.legend()
fig.canvas.draw()
writer.add_image(f"Rest/TSNE", grab_buffer(fig), dataformats='HWC')
fig.savefig("test.png")
exit()

midi_notes = range(21,109)
abs_diffs = []

for i in midi_notes:
    z_predict = disentangle.get_new_sample(z_codes[0].unsqueeze(0), torch.tensor(i).unsqueeze(0).to(device), p_start=p_old[0].unsqueeze(0), mfcc=mfcc[0].unsqueeze(0), rms=rms[0].unsqueeze(0), inst=inst[0].unsqueeze(0))
    z_predict = model.quantizer.from_codes(z_predict)[0]
    out = model.decode(z_predict)

    pi = torch.tensor([i]).unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float().to(device)
    pi[:,:, ((z.shape[-1]*disentangle.pc_num)//disentangle.pc_denom):] = torch.tensor(0).to(disentangle.device)

    p_predict = disentangle.pitch_predict(z_predict)
    p_predict = torch.argmax(p_predict, dim=1).unsqueeze(1)
    abs_diffs.append(torch.mean(torch.abs(pi - p_predict)).cpu().detach().numpy())

    # writer.add_audio(f"Audio/Output-{i}:"  , out[0])
    # writer.add_audio(f"Audio/Predict: pitch = {pitches[i]}" , out[0])
fig, ax = plt.subplots()

ax.plot(list(midi_notes), abs_diffs)
ax.set(title='Absolute difference of predict pitches')
ax.set_xlabel("Midi note")
ax.set_ylabel("Abs Difference")

fig.canvas.draw()
writer.add_image(f"Pitch/Sweep", grab_buffer(fig), dataformats='HWC')

# l_curve1 = torch.arange(0,1,1/z.shape[-1])
# l_curve2 = torch.arange(1,0,1/-z.shape[-1])
# l_curve3 = torch.cat((torch.arange(0,1,1/(z.shape[-1]/2)), torch.arange(1,0,-1/(z.shape[-1]/2))))[:z.shape[-1]]
# l_curve4 = torch.cat((torch.arange(0,1,1/(3*z.shape[-1]/4)), torch.arange(1,0,-1/(z.shape[-1]/4))))[:z.shape[-1]]

# curves = [l_curve1, l_curve2, l_curve3,l_curve4]

# for i in range(4):
#     z_predict = disentangle.get_new_sample(z_codes[0].unsqueeze(0), rms=curves[i].unsqueeze(0).unsqueeze(0).to(device), mfcc=mfcc[0].unsqueeze(0), p=p_old[0].unsqueeze(0))
#     z_predict = model.quantizer.from_codes(z_predict)[0]
#     out = model.decode(z_predict)

#     rms_predict = disentangle.rms_predict(z_predict)

#     # writer.add_audio(f"Audio/Output-{i}:"  , out[0])
#     writer.add_audio(f"Audio/Predict: loudness = curve {i+1}" , out[0])

#     writer.add_image(f"RMS/Loudness Curve {i}", make_rms_img(curves[i].unsqueeze(0).unsqueeze(0).to(device)), dataformats='HWC')
#     writer.add_image(f"RMS/Predict Loudness Curve {i}",make_rms_img(rms_predict), dataformats='HWC')



# def grab_buffer(fig):
#     data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     return data

# def make_pyin_img(samples, pyin):
#     f0 = pyin[0].cpu().detach().numpy()
#     times = librosa.times_like(f0)

#     y = samples.cpu().detach().numpy()


#     D = librosa.amplitude_to_db(np.abs(librosa.stft(y.T)), ref=np.max)
#     fig, ax = plt.subplots()
#     img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
#     ax.set(title='pYIN fundamental frequency estimation')
#     fig.colorbar(img, ax=ax, format="%+2.f dB")
#     ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
#     ax.legend(loc='upper right')

#     fig.canvas.draw()

#     return grab_buffer(fig)

# def make_mfcc_img(mfcc):
#     mfcc = mfcc[0].cpu().detach().numpy()

#     fig, ax = plt.subplots()
#     img = librosa.display.specshow(mfcc, x_axis='time', ax=ax)
#     fig.colorbar(img, ax=ax)
#     ax.set(title='MFCC')

#     fig.canvas.draw()

#     return grab_buffer(fig)

# def make_rms_img(rms):
#     rms = rms[0].cpu().detach().numpy()
    
#     fig, ax = plt.subplots()
#     times = librosa.times_like(rms)
#     ax.semilogy(times, rms[0], label='RMS Energy')
#     ax.set(xticks=[])
#     ax.legend()
#     ax.label_outer()

#     fig.canvas.draw()

#     return grab_buffer(fig)
    

# sr = 44100

# #data setup
# print("LOADING VALIDATION DATA...")
# data_path = ["valid_tensor.pt"]#"/vast/df2322/data/Nsynth/nsynth-valid"
# data = dataset.NSynth_ram(data_path)
# # d_path = ["train_tensor_" + str(i) + ".pt" for i in range(12)]
# # data = dataset.NSynth_ram(d_path)
# valid_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)
# print("DONE!!")

# #set device used to perform training
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# #log for tensorboard
# writer = SummaryWriter("tensorboard/inference_runs")

# #load saved models
# c_enc = torch.load("saved_models/disentangle_c_enc.pt",map_location=device).to(device)
# t_enc = torch.load("saved_models/disentangle_t_enc.pt",map_location=device).to(device)
# p_enc = torch.load("saved_models/disentangle_p_enc.pt",map_location=device).to(device)
# l_enc = torch.load("saved_models/disentangle_l_enc.pt",map_location=device).to(device)
# dec = torch.load("saved_models/disentangle_dec.pt",map_location=device).to(device)

# #create DAC encoder and decoder
# model_path = dac.utils.download(model_type="44khz")
# model = dac.DAC.load(model_path).to(device)
# model.eval()

# for i in range(3):
#     # samples,mfcc,pyin, rms = next(iter(valid_loader))

#     # samples = samples.to(device)
#     # mfcc = mfcc.to(device)
#     # pyin = pyin.to(device)
#     # rms = rms.to(device)

#     # z, codes, latents, _, _ = model.encode(samples[:,None,:])

#     # c_emb, _, vq_losses = c_enc(z)
#     # t_emb = t_enc(z)
#     # p_emb = p_enc(z)
#     # l_emb = l_enc(z)

#     # emb = torch.cat((t_emb, p_emb, l_emb, c_emb), 1)
#     # z_rec = dec(emb)

#     # output_audio = model.decode(z_rec)

#     # writer.add_audio(f"Audio/Input-{i}:"  , samples)
#     # writer.add_audio(f"Audio/Reconstruction-{i}" , output_audio[0])

#     # samples = samples[0]
#     # pyin =  (pyin * (data.pitch_max - data.pitch_min) - data.pitch_min)[0]
#     # p_pred = (p_emb * (data.pitch_max - data.pitch_min) - data.pitch_min)[0]
#     # mfcc =  mfcc * (data.mfcc_max - data.mfcc_min) - data.mfcc_min
#     # t_pred = (t_emb * (data.mfcc_max - data.mfcc_min) + data.mfcc_min)


#     # writer.add_image(f"Pitch/Input-{i}", make_pyin_img(samples, pyin), dataformats='HWC')
#     # writer.add_image(f"Pitch/Reconstruction-{i}", make_pyin_img(samples, p_pred), dataformats='HWC')

#     # writer.add_image(f"MFCC/Input-{i}", make_mfcc_img(mfcc), dataformats='HWC')
#     # writer.add_image(f"MFCC/Reconstruction-{i}", make_mfcc_img(t_pred), dataformats='HWC')

#     # writer.add_image(f"RMS/Input-{i}", make_rms_img(rms), dataformats='HWC')
#     # writer.add_image(f"RMS/Reconstruction-{i}", make_rms_img(l_emb), dataformats='HWC')
#     z,mfcc,pyin, rms = next(iter(valid_loader))

#     z = z.to(device)[:,0,:,:]
#     mfcc = mfcc.to(device)
#     pyin = pyin.to(device)
#     rms = rms.to(device)

#     c_emb, vq_losses = c_enc(z)
#     t_emb = t_enc(z)
#     p_emb = p_enc(z)
#     l_emb = l_enc(z)

#     emb = torch.cat((t_emb, p_emb, l_emb, c_emb), 1)

#     z_rec = dec(emb)

#     z = (z * (data.z_max - data.z_min) + data.z_min)
#     z_rec = (z_rec * (data.z_max - data.z_min) + data.z_min)

#     input_audio = model.decode(z)
#     output_audio = model.decode(z_rec)

#     writer.add_audio(f"Audio/Input-{i}:"  , input_audio[0])
#     writer.add_audio(f"Audio/Reconstruction-{i}" , output_audio[0])

#     samples = input_audio[0][0]
#     pyin =  (pyin * (data.pitch_max - data.pitch_min) + data.pitch_min)[0]
#     p_pred = (p_emb * (data.pitch_max - data.pitch_min) + data.pitch_min)[0]
#     mfcc =  mfcc * (data.mfcc_max - data.mfcc_min) + data.mfcc_min
#     t_pred = (t_emb * (data.mfcc_max - data.mfcc_min) + data.mfcc_min)


#     writer.add_image(f"Pitch/Input-{i}", make_pyin_img(samples, pyin), dataformats='HWC')
#     writer.add_image(f"Pitch/Reconstruction-{i}", make_pyin_img(samples, p_pred), dataformats='HWC')

#     writer.add_image(f"MFCC/Input-{i}", make_mfcc_img(mfcc), dataformats='HWC')
#     writer.add_image(f"MFCC/Reconstruction-{i}", make_mfcc_img(t_pred), dataformats='HWC')

#     writer.add_image(f"RMS/Input-{i}", make_rms_img(rms), dataformats='HWC')
#     writer.add_image(f"RMS/Reconstruction-{i}", make_rms_img(l_emb), dataformats='HWC')

# writer.flush()
# writer.close()