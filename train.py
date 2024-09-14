import dataset
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
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchmetrics.clustering import MutualInfoScore
import matplotlib.pyplot as plt
import random

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


def main(rank, world_size):
    import model

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


    #get training and validation datasets
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = rank
    if rank == 0:
        print("LOADING TRAINING DATA...")
    # d_path = ["train_tensor_" + str(i) + ".pt" for i in range(12)]
    # d_path = ["test_tensor_JC_" + str(i) + ".pt" for i in range(1)]
    data = dataset.NSynth_transform_ram(training_params["data_path"], instruments=training_params["instruments"])
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=DistributedSampler(data), drop_last=False, num_workers=training_params["num_workers"])
    if rank == 0:
        print("DONE!!")

    valid_loader = train_loader

    if rank == 0:
        print("LOADING VALIDATION DATA...")
    # d_path = ["train_tensor_" + str(i) + ".pt" for i in range(12)]
    # v_d_path = ['valid_tensor_JC_0.pt', 'valid_tensor_JC_1.pt']
    v_data = dataset.NSynth_transform_ram(training_params["validation_data_path"], instruments=training_params["instruments"])
    valid_loader = torch.utils.data.DataLoader(v_data, batch_size=5, sampler=DistributedSampler(data), drop_last=False, num_workers=training_params["num_workers"])
    if rank == 0:
        print("DONE!!")

    v_frequency = training_params["v_frequency"]


    disentangle = model.disentangle(device=rank).to(device)
    disentangle = DDP(disentangle, device_ids=[device], output_device=rank, find_unused_parameters=True)

    # Initialize optimizer.
    lr = training_params["lr"]

    optimizer = optim.Adam([p for name, p in disentangle.module.named_parameters() if "pitch_predict" not in name and "mfcc_predict" not in name and "rms_predict" not in name and "dacModel" not in name], lr=lr)
    rest_parameters = [name for name, p in disentangle.module.named_parameters() if "pitch_predict" not in name and "mfcc_predict" not in name and "rms_predict" not in name and "dacModel" not in name]
    
    
    #optimizer for feature vectors
    feature_optimizer = optim.Adam([p for name, p in disentangle.module.named_parameters() if "pitch_predict" in name or "mfcc_predict" in name or "rms_predict" in name ], lr=lr)
    enc_parameters = [name for name, p in disentangle.module.named_parameters() if "pitch_predict" in name or "mfcc_predict" in name or "rms_predict" in name ]
    #add learning rate decay
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, gamma=0.99)

    #log for tensorboard
    writer = SummaryWriter(training_params["tb_path"])

    if training_params["train_continue"]:
        old_model = torch.load(save_path).to(device)
        disentangle = disentangle.module
        disentangle.load_state_dict(old_model.state_dict())
        disentangle = DDP(disentangle, device_ids=[device], output_device=rank, find_unused_parameters=True)
        optimizer = optim.Adam([p for name, p in disentangle.module.named_parameters() if "pitch_predict" not in name and "mfcc_predict" not in name and "rms_predict" not in name and "dacModel" not in name], lr=lr)
        feature_optimizer = optim.Adam([p for name, p in disentangle.module.named_parameters() if "pitch_predict" in name or "mfcc_predict" in name or "rms_predict" in name ], lr=lr)
        torch.distributed.barrier()

    if training_params["train_encoder"]:
        epochs = training_params["encoder_epochs"]
        for epoch in range(1, epochs+1):
            disentangle.train()
            train_loader.sampler.set_epoch(epoch)

            loss_total_pitch = 0
            loss_total_mfcc = 0
            loss_total_rms = 0

            for (batch_idx, train_tensors) in enumerate(train_loader):
                z,p,mfcc,rms,z_prime,p_prime,mfcc_prime,rms_prime = train_tensors   

                z = z.to(device)[:,0,:,:]
                p = p.to(device)
                mfcc = mfcc.to(device)
                rms = rms.to(device)
                l,_ = disentangle.module.train_enc(z,p,mfcc, rms)

                loss = l["p_predict"] + l["m_predict"] + l["r_predict"]
                loss.sum().backward()
                torch.nn.utils.clip_grad_norm_(disentangle.parameters(), 0.0001)

                feature_optimizer.step()
                feature_optimizer.zero_grad(set_to_none=True)

                loss_total_pitch +=  l["p_predict"].item() 
                loss_total_mfcc += l["m_predict"].item()
                loss_total_rms += l["r_predict"].item()

                if training_params["verbos"]:
                    if rank == 0:
                        print(f"sample {batch_idx + 1} out of {len(train_loader)}")
                        print(loss.item())
            
            writer.add_scalar("Encoder Loss/Pitch Predict", loss_total_pitch / len(train_loader), epoch)
            writer.add_scalar("Encoder Loss/Timbre Predict", loss_total_mfcc / len(train_loader), epoch)
            writer.add_scalar("Encoder Loss/Loudness Predict", loss_total_rms / len(train_loader), epoch)
    else:
        old_model = torch.load(save_path).to(device)
        disentangle = disentangle.module
        disentangle.pitch_predict.load_state_dict(old_model.pitch_predict.state_dict())
        disentangle.mfcc_predict.load_state_dict(old_model.mfcc_predict.state_dict())
        disentangle.rms_predict.load_state_dict(old_model.rms_predict.state_dict())
        disentangle = DDP(disentangle, device_ids=[device], output_device=rank, find_unused_parameters=True)
        optimizer = optim.Adam([p for name, p in disentangle.module.named_parameters() if "pitch_predict" not in name and "mfcc_predict" not in name and "rms_predict" not in name and "dacModel" not in name], lr=lr)
        feature_optimizer = optim.Adam([p for name, p in disentangle.module.named_parameters() if "pitch_predict" in name or "mfcc_predict" in name or "rms_predict" in name ], lr=lr)
        torch.distributed.barrier()

    # epochs = training_params["epochs"]
    # for epoch in range(1, epochs+1):
    #     disentangle.train()
    #     train_loader.sampler.set_epoch(epoch + training_params["encoder_epochs"])

    #     loss_total_r = 0

    #     for (batch_idx, train_tensors) in enumerate(train_loader):
            
    #         z,p,mfcc,rms,z_prime,p_prime,mfcc_prime,rms_prime = train_tensors   

    #         z = z.to(device)[:,0,:,:]
    #         z_prime = z_prime.to(device)[:,0,:,:]
    #         p_prime = p.to(device)
    #         mfcc_prime = mfcc_prime.to(device)
    #         rms_prime = rms_prime.to(device)
    #         l,_ = disentangle.module.train_recon(z,z_prime, p_prime,mfcc_prime, rms_prime)

    #         loss = l["t_predict"] 
            

    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(disentangle.parameters(), 0.0001)

    #         optimizer.step()
    #         optimizer.zero_grad(set_to_none=True)

    #         loss_total_r +=  loss

    #         if training_params["verbos"]:
    #             if rank == 0:
    #                 print(f"sample {batch_idx + 1} out of {len(train_loader)}")
    #                 print(loss.item())
        
    #     writer.add_scalar("Loss/Recon", loss_total_r / len(train_loader), epoch)

    epochs = training_params["epochs"]
    for epoch in range(1,epochs+1): 
        disentangle.train()

        train_loader.sampler.set_epoch(epoch + training_params["encoder_epochs"])

        loss_total = 0
        loss_total_pitch = 0
        loss_total_mfcc = 0
        loss_total_rms = 0
        loss_total_r = 0
        loss_total_ce = 0

        for (batch_idx, train_tensors) in enumerate(train_loader):
            z,p,mfcc,rms,inst,z_prime,p_prime,mfcc_prime,rms_prime,inst_prime = train_tensors   

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
            l,_ = disentangle(z,p,mfcc,rms,inst,z_prime, p_prime, mfcc_prime, rms_prime, inst_prime)
            
            # if epoch < training_params["encoder_epochs"]:
            #     loss = l["p_predict"] + l["m_predict"] + l["r_predict"]
            # else:
            #     loss = l["t_predict"] + l["p_ce_loss"] + l["m_ce_loss"] + l["r_ce_loss"] + l["pm_ce_loss"] + l ["pr_ce_loss"] + l["mr_ce_loss"]

            # if epoch < training_params["encoder_epochs"]:
            #     loss = 0.0*l["t_predict"] + 0.0*l["p_ce_loss"] + 0.0*l["r_ce_loss"] + 0.0*l["pr_ce_loss"] + l["p_predict"] + l["m_predict"] + l["r_predict"]
            # else:

            loss = l["t_predict"] + l["p_ce_loss"] #+ l["r_ce_loss"] + l["pr_ce_loss"]


            loss.sum().backward()
            torch.nn.utils.clip_grad_norm_(disentangle.parameters(), 0.0001)

            # if epoch < training_params["encoder_epochs"]:
            #     feature_optimizer.step()
            #     feature_optimizer.zero_grad(set_to_none=True)
            # else:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # if epoch == training_params["encoder_epochs"]:
                
            #     feature_optimizer.step()
            #     feature_optimizer.zero_grad(set_to_none=True)
                # disentangle = disentangle.module
                # disentangle.stop_encoder_training()
                
                # for param in disentangle.parameters():
                #     param.requires_grad = False

                # for param in disentangle.encode_pitch.parameters():
                #     param.requires_grad = True

                # for param in disentangle.encode_mfcc.parameters():
                #     param.requires_grad = True

                # for param in disentangle.encode_rms.parameters():
                #     param.requires_grad = True

                # for param in disentangle.encode_rms.parameters():
                #     param.requires_grad = True

                # for param in disentangle.decoder.parameters():
                #     param.requires_grad = True

                # disentangle.pitch_predict.eval()
                # disentangle.mfcc_predict.eval()
                # disentangle.rms_predict.eval()
                # disentangle.dacModel.eval()

                # disentangle = DDP(disentangle, device_ids=[device], output_device=rank, find_unused_parameters=True)
                # optimizer = optim.Adam( filter(lambda p: p.requires_grad, disentangle.module.parameters()),lr=lr)
            

            loss_total += loss.item()

            # loss_total_pitch +=  l["p_recon"].item() 
            # loss_total_mfcc += l["m_recon"].item()
            # loss_total_rms += l["r_recon"].item()s
            loss_total_r += l["t_predict"].item()
            loss_total_ce += (l["p_ce_loss"]).item() #+ l["r_ce_loss"] + l["pr_ce_loss"]).item()

            if training_params["verbos"]:
                if rank == 0:
                    print(f"sample {batch_idx + 1} out of {len(train_loader)}")
                    print(loss.item())

        
        # scheduler.step()
        # print(scheduler.get_last_lr())
        writer.add_scalar("Loss/Total", loss_total / len(train_loader), epoch)
        writer.add_scalar("Loss/Recon", loss_total_r / len(train_loader), epoch)
        # writer.add_scalar("Loss/Pitch Predict", loss_total_pitch / len(train_loader), epoch)
        # writer.add_scalar("Loss/Timbre Predict", loss_total_mfcc / len(train_loader), epoch)
        # # writer.add_scalar("Loss/Loudness Predict", loss_total_rms / len(train_loader), epoch)
        # writer.add_scalar("Loss/Pitch Recon", loss_total_pitch / len(train_loader), epoch)
        # writer.add_scalar("Loss/Timbre Recon", loss_total_mfcc / len(train_loader), epoch)
        # writer.add_scalar("Loss/Loudness Recon", loss_total_rms / len(train_loader), epoch)
        writer.add_scalar("Loss/Cosine Embedding", loss_total_ce / len(train_loader), epoch)

        if (epoch % v_frequency) == 0:
            #evaluate mode
            disentangle.eval()

            if rank == 0:
                torch.save(disentangle.module, save_path)

            v_loss_total = 0
            # v_loss_total_pitch = 0
            # v_loss_total_mfcc = 0
            # v_loss_total_rms = 0
            v_loss_total_r = 0
            v_loss_total_ce = 0

            i = 0
            valid_loader.sampler.set_epoch(0)
            for v_data in valid_loader:
                #create DAC encoder and decoder
                model_path = dac.utils.download(model_type="44khz")
                model = dac.DAC.load(model_path).to(device)
                model.eval()

                z,p,mfcc,rms,inst,z_prime,p_prime,mfcc_prime,rms_prime,inst_prime = v_data 

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
                l,predict = disentangle(z,p,mfcc,rms,inst,z_prime, p_prime, mfcc_prime, rms_prime, inst_prime)

                # v_loss_total_pitch +=  l["p_predict"].item()
                # v_loss_total_mfcc += l["m_predict"].item()
                # v_loss_total_rms += l["r_predict"].item()
                v_loss_total_r += l["t_predict"].item()
                v_loss_total_ce += (l["p_ce_loss"]).item() #+ l["r_ce_loss"] + l["pr_ce_loss"]).item()
                v_loss_total = v_loss_total_r + v_loss_total_ce #+ v_loss_total_pitch + v_loss_total_mfcc + v_loss_total_rms 

                #make p span across time
                p = p.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
                p[:,:, ((z.shape[-1]*disentangle.module.pc_num)//disentangle.module.pc_denom):] = torch.tensor(0).to(disentangle.module.device)

                p_prime = p_prime.unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float()
                p_prime[:,:, ((z.shape[-1]*disentangle.module.pc_num)//disentangle.module.pc_denom):] = torch.tensor(0).to(disentangle.module.device)

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

                writer.add_audio(f"Audio/Ground Truth-{i}:"  , input_audio[0])
                writer.add_audio(f"Audio/Reconstruction-{i}" , output_audio[0])

                samples = input_audio[0][0]
                mfcc =  mfcc_prime * (data.mfcc_max - data.mfcc_min) + data.mfcc_min
                mfcc_hat = (predict["mfcc"] * (data.mfcc_max - data.mfcc_min) + data.mfcc_min)


                writer.add_image(f"Pitch/Ground Truth-{i}", make_pitch_img(samples, p_prime[0]), dataformats='HWC')
                writer.add_image(f"Pitch/Reconstruction-{i}", make_pitch_img(samples, p_hat[0]), dataformats='HWC')

                writer.add_image(f"MFCC/Ground Truth-{i}", make_mfcc_img(mfcc), dataformats='HWC')
                writer.add_image(f"MFCC/Reconstruction-{i}", make_mfcc_img(mfcc_hat), dataformats='HWC')

                writer.add_image(f"RMS/Ground Truth-{i}", make_rms_img(rms_prime), dataformats='HWC')
                writer.add_image(f"RMS/Reconstruction-{i}", make_rms_img(predict["rms"]), dataformats='HWC')
                if i >2:
                    break
                i += 1
            
            writer.add_scalar("Validation Loss/Total", v_loss_total / len(valid_loader), epoch)
            writer.add_scalar("Validation Loss/Recon", v_loss_total_r / len(valid_loader), epoch)
            # writer.add_scalar("Validation Loss/Pitch Predict", v_loss_total_pitch / len(valid_loader), epoch)
            # writer.add_scalar("Validation Loss/Timbre Predict", v_loss_total_mfcc / len(valid_loader), epoch)
            # writer.add_scalar("Validation Loss/Loudness Predict", v_loss_total_rms / len(valid_loader), epoch)
            writer.add_scalar("Validation Loss/Cosine Embedding", v_loss_total_ce / len(valid_loader), epoch)
            
            #create DAC encoder and decoder
            model_path = dac.utils.download(model_type="44khz")
            model = dac.DAC.load(model_path).to(device)
            model.eval()
        
            z,p,mfcc,rms,inst,z_prime,p_prime,mfcc_prime,rms_prime,inst_prime = next(iter(valid_loader))

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
            
            z_hat = predict["z"]

            #convert from discrete code to continuous embedding
            z_codes = z
            z = model.quantizer.from_codes(z)[0]


            input_audio = model.decode(z[0].unsqueeze(0))
            writer.add_audio(f"Audio/Input:"  , input_audio[0])

            pitches = torch.tensor([num for num in [61, 56, 26, 35]]).to(device)
            
            for i in range(4):
                z_predict = disentangle.module.get_new_sample(z_codes[0].unsqueeze(0), pitches[i].unsqueeze(0), p_start=p[0].unsqueeze(0), mfcc=mfcc[0].unsqueeze(0), rms=rms[0].unsqueeze(0), inst=inst[0].unsqueeze(0))
                z_predict = model.quantizer.from_codes(z_predict)[0]
                out = model.decode(z_predict)

                # writer.add_audio(f"Audio/Output-{i}:"  , out[0])
                writer.add_audio(f"Audio/Predict: pitch = {pitches[i]}" , out[0])

            # l_curve1 = torch.arange(0,1,1/z.shape[-1])
            # l_curve2 = torch.arange(1,0,1/-z.shape[-1])
            # l_curve3 = torch.cat((torch.arange(0,1,1/(z.shape[-1]/2)), torch.arange(1,0,-1/(z.shape[-1]/2))))[:z.shape[-1]]
            # l_curve4 = torch.cat((torch.arange(0,1,1/(3*z.shape[-1]/4)), torch.arange(1,0,-1/(z.shape[-1]/4))))[:z.shape[-1]]

            # curves = [l_curve1, l_curve2, l_curve3,l_curve4]
            # for i in range(4):
            #     z_predict = disentangle.module.get_new_sample(z_codes[0].unsqueeze(0), rms=curves[i].unsqueeze(0).unsqueeze(0).to(device), mfcc=mfcc[0].unsqueeze(0), p=p[0].unsqueeze(0))
            #     z_predict = model.quantizer.from_codes(z_predict)[0]
            #     out = model.decode(z_predict)

            #     # writer.add_audio(f"Audio/Output-{i}:"  , out[0])
            #     writer.add_audio(f"Audio/Predict: loudness = curve {i+1}" , out[0])

            # total_mi_score = 0
            # for (batch_idx, valid_tensors) in enumerate(valid_loader):
            #     z,p,z_prime,p_prime = train_tensors

            #     z = z.to(device)[:,0,:,:]
            #     p = p.to(device)
            #     z_prime = z_prime.to(device)[:,0,:,:]
            #     p_prime = p_prime.to(device)

            #     _,_,_ = disentangle(z,p,z_prime,p_prime)

            #     pe = disentangle.module.pitch_emb
            #     re = disentangle.module.rest_emb


            #     mi_score = MutualInfoScore()
            #     total_mi_score += mi_score
            #     score = mi_score(torch.flatten(pe.int()),torch.flatten(re.int()))

            # writer.add_scalar("Metric/Mutal Information P and R", total_mi_score.compute() / len(valid_loader), epoch)

        
    ddp_cleanup()
    # #evaluate mode
    # disentangle.eval()

    # #create DAC encoder and decoder
    # model_path = dac.utils.download(model_type="44khz")
    # model = dac.DAC.load(model_path).to(device)
    # model.eval()

    # z,p,mfcc,rms,z_prime = next(iter(train_loader))

    # z = z.to(device)[:,0,:,:]
    # p = p.to(device)
    # mfcc = mfcc.to(device)
    # z_prime = z_prime.to(device)[:,0,:,:]

    # _,predict = disentangle(z,p,mfcc, rms,z_prime)
    
    # z_hat = predict["z"]

    # #convert from discrete code to continuous embedding
    # z_codes = z
    # z = model.quantizer.from_codes(z)[0]


    # input_audio = model.decode(z[0].unsqueeze(0))
    # writer.add_audio(f"Audio/Input:"  , input_audio[0])

    # pitches = torch.tensor([num for num in [61, 56, 26, 26, 35]]).to(device)
    # for i in range(5):
    #     z_predict = disentangle.module.get_new_sample(z_codes[0].unsqueeze(0), pitches[i].unsqueeze(0))
    #     z_predict = model.quantizer.from_codes(z_predict)[0]
    #     out = model.decode(z_predict)

    #     # writer.add_audio(f"Audio/Output-{i}:"  , out[0])
    #     writer.add_audio(f"Audio/Predict: pitch = {pitches[i]}" , out[0])

    # pitches = [torch.tensor(num).to(device).unsqueeze(0) for num in [61, 56, 26, 26, 35]]
    # for i in range(5):
    #     # zin = (zin * (data.z_max - data.z_min) + data.z_min)
    #     # z_prime[i] = (z_prime[i] * (data.z_max - data.z_min) + data.z_min)
    #     # z_prime_hat[i] = (z_prime_hat[i] * (data.z_max - data.z_min) + data.z_min)

    #     # input_audio = model.decode(z[i].unsqueeze(0))
    #     # ground_truth_audio = model.decode(z_prime[i].unsqueeze(0))
    #     if i == 4:
    #         z_predict = disentangle.module(z_codes[i].unsqueeze(0), p[i].unsqueeze(0),z_codes[i].unsqueeze(0), p[i].unsqueeze(0))[3]

    #     else:
    #         z_predict = disentangle.module.get_new_pitch(z_codes[0].unsqueeze(0), pitches[i])
    #     # z_predict = model.quantizer.from_codes(z_predict)[0]
    #     predict_audio = model.decode(z_predict)

    #     # writer.add_audio(f"Audio/Input-{i}:"  , input_audio[0])
    #     # writer.add_audio(f"Audio/Ground Truth-{i}" , ground_truth_audio[0])
    #     writer.add_audio(f"Audio/Predict: pitch = {pitches[i][0]}" , predict_audio[0])

    writer.flush()
    writer.close()








    # # Train model.
    # eval_every = 1
    # best_train_loss = float("inf")
    # timbre_enc.train()
    # pitch_enc.train()
    # loudness_enc.train()
    # content_enc.train()
    # decoder.train()





if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)