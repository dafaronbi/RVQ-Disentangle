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

        loss = recon_loss + mfcc_loss + pitch_loss + loudness_loss + commitment_loss + codebook_loss
        loss.sum().backward()
        optimizer.step()

        total_train_loss += loss.sum().item()
        total_recon_loss += recon_loss.item()
        total_mfcc_loss += mfcc_loss.item()
        total_pitch_loss += pitch_loss.item()
        total_loudness_loss += loudness_loss.item()
        total_commitment_loss += commitment_loss.sum().item()
        total_codebook_loss += codebook_loss.sum().item()

        print(f"Total Loss: {loss.sum().item()}")
        print(f"Reconstruction Loss: {recon_loss.item()}")
        print(f"Tibmre Loss: {mfcc_loss.item()}")
        print(f"Pitch Loss: {pitch_loss.item()}")
        print(f"Loudness Loss: {loudness_loss.item()}")
        print(f"Commitment Loss: {commitment_loss.sum().item()}")
        print(f"Codebook {codebook_loss.sum().item()}")

    writer.add_scalar("Loss/Total", total_train_loss, epoch)
    writer.add_scalar("Loss/recon", total_recon_loss, epoch)
    writer.add_scalar("Loss/mfcc", total_mfcc_loss, epoch)
    writer.add_scalar("Loss/pitch", total_pitch_loss, epoch)
    writer.add_scalar("Loss/loudness", total_loudness_loss, epoch)
    writer.add_scalar("Loss/commitmentl", total_commitment_loss, epoch)
    writer.add_scalar("Loss/codebook", total_codebook_loss, epoch)

    if  epoch % 2 == 0:
        if total_train_loss < best_train_loss:
                best_train_loss = total_train_loss
                torch.save(timbre_enc, training_params["save_path"] + "_t_enc.pt")
                torch.save(pitch_enc, training_params["save_path"] + "_p_enc.pt")
                torch.save(loudness_enc, training_params["save_path"] + "_l_enc.pt")
                torch.save(content_enc, training_params["save_path"] + "_c_enc.pt")
                torch.save(decoder, training_params["save_path"] + "_dec.pt")
    

# Generate and save reconstructions.
# network.eval()

# audio = next(iter(train_loader))
# out = network(audio[0].to(device))

# writer.add_audio("Input Audio "  , audio[0][0], 0, 16000)
# writer.add_audio("Reconstuction " , out["x_recon"][0], 0, 16000)

writer.flush()
writer.close()