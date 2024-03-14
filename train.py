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
batch_size = training_params["batch_size"]

#log for tensorboard
writer = SummaryWriter(training_params["tb_path"])


# #get training and validation datasets

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = 'cpu'


dataset = dataset.NSynth(
        data_path,
        transforms = [librosa.feature.mfcc, librosa.pyin, librosa.feature.rms],
        blacklist_pattern=["string"],  # blacklist string instrument
        categorical_field_list=["instrument_family", "instrument_source"])
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#create style encoders
timbre_enc = model.style_enc(20).to(device)
pitch_enc = model.style_enc(1).to(device)
loudness_enc = model.style_enc(1).to(device)

#create content enc
content_enc = model.content_enc().to(device)

#create decoder
decoder = model.decoder(1046).to(device)

model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path).to(device)
model.eval()



# Initialize optimizer.
lr = training_params["lr"]
optimizer = optim.Adam(list(timbre_enc.parameters()) + list(pitch_enc.parameters()) + list(loudness_enc.parameters()) 
+ list(content_enc.parameters()) + list(decoder.parameters()), lr=lr)
# criterion = auraloss.freq.MultiResolutionSTFTLoss(
#     fft_sizes=[1024, 2048, 8192],
#     hop_sizes=[256, 512, 2048],
#     win_lengths=[1024, 2048, 8192],
#     scale="mel",
#     n_bins=128,
#     sample_rate=sr,
#     perceptual_weighting=True,
# )
criterion = torch.nn.MSELoss()

# Train model.
epochs = 7
eval_every = 1
beta = training_params["beta"]
best_train_loss = float("inf")
timbre_enc.train()
pitch_enc.train()
loudness_enc.train()
content_enc.train()
decoder.train()
for epoch in range(training_params["epochs"]):
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
        pitch = pitch.to(device)
        rms = rms.to(device)

        z, codes, latents, _, _ = model.encode(samples[:,None,:])
        
        t_emb = timbre_enc(z)
        p_emb = pitch_enc(z)
        l_emb = loudness_enc(z)
        c_emb, _, vq_losses = content_enc(z)

        commitment_loss = vq_losses["commitment"]
        codebook_loss = vq_losses["codebook"]

        mfcc_loss = criterion(nn.functional.pad(t_emb, (0,max(mfcc.shape[-1] - t_emb.shape[-1],0)), "constant",0), mfcc)
        pitch_loss = criterion(nn.functional.pad(p_emb, (0,max(pitch.shape[-1] - p_emb.shape[-1],0)), "constant",0), pitch)
        loudness_loss = criterion(nn.functional.pad(l_emb, (0,max(rms.shape[-1] - l_emb.shape[-1],0)), "constant",0), rms)

        emb = torch.cat((t_emb, p_emb, l_emb, c_emb), 1)

        z_rec = decoder(emb)
        recon_loss = criterion(z, z_rec)

        loss = recon_loss + mfcc_loss + pitch_loss + loudness_loss + commitment_loss + codebook_loss
        loss.mean().backward()
        optimizer.step()

        total_train_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_mfcc_loss += mfcc_loss.item()
        total_pitch_loss += pitch_loss.item()
        total_loudness_loss += loudness_loss.item()
        total_commitment_loss += commitment_loss.item()
        total_codebook_loss += codebook_loss.item()
        print("here")

    
    print(f"Total Loss: {total_train_loss}")
    print(f"Reconstruction Loss: {total_recon_loss}")
    print(f"Tibmre Loss: {total_mfcc_loss}")
    print(f"Pitch Loss: {total_pitch_loss}")
    print(f"Loudness Loss: {total_loudness_loss}")
    print(f"Commitment Loss: {total_commitment_loss}")
    print(f"Codebook {total_codebook_loss}")

        # optimizer.zero_grad()
        # audio = train_tensors[0].to(device)
        # out = network(audio)
        # recon_error = criterion(out["x_recon"], audio)
        # total_recon_error += recon_error.item()
        # loss = recon_error + beta * out["commitment_loss"]
        # if not training_params["use_ema"]:
        #     loss += out["dictionary_loss"]

        # total_train_loss += loss.item()
        # loss.backward()
        # optimizer.step()
        # n_train += 1

        # if ((batch_idx + 1) % eval_every) == 0:
        #     print(f"epoch: {epoch}\nbatch_idx: {batch_idx + 1}", flush=True)
        #     total_train_loss /= n_train
        #     if total_train_loss < best_train_loss:
        #         best_train_loss = total_train_loss
        #         torch.save(network, training_params["save_path"])

        #     print(f"total_train_loss: {total_train_loss}")
        #     print(f"best_train_loss: {best_train_loss}")
        #     print(f"recon_error: {total_recon_error / n_train}\n")

        #     writer.add_scalar("Loss/Total", total_train_loss, epoch)
        #     writer.add_scalar("Loss/Reconstruction", total_recon_error, epoch)

        #     total_train_loss = 0
        #     total_recon_error = 0
        #     n_train = 0
    

# Generate and save reconstructions.
# network.eval()

# audio = next(iter(train_loader))
# out = network(audio[0].to(device))

# writer.add_audio("Input Audio "  , audio[0][0], 0, 16000)
# writer.add_audio("Reconstuction " , out["x_recon"][0], 0, 16000)

# writer.flush()
# writer.close()