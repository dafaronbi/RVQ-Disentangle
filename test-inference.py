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
import random
import argparse
import datetime
import sklearn
from sklearn.cluster import KMeans


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

parser = argparse.ArgumentParser(description='Data Directory')
parser.add_argument('-e', '--experiments', help="experiments to run", default="")
parser.add_argument('-m', '--model', help="model to test", default="saved_models/disentangle.pt")
parser.add_argument('-d', '--dataset', help="dataset to run inference", type=lambda s: [item for item in s.split(' ')], default=["data/test_tensor_JC_0.pt"])
parser.add_argument('-i', '--instruments', help="instruments of dataset", type=lambda s: [int(item) for item in s.split(' ')], default=None)
args = parser.parse_args()


#get training and validation datasets
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
gpu_count = torch.cuda.device_count()

# disentangle = model.disentangle(device=device).to(device)
disentangle = torch.load(args.model, map_location=device).to(device)
# disentangle.load_state_dict(torch.load("saved_models/BEST_disentangle.pt").state_dict()) 
disentangle.device = device
disentangle.eval()

# print(disentangle)
# exit()

#log for tensorboard
writer = SummaryWriter("tensorboard/inference_runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

data = dataset.NSynth_transform_ram(args.dataset,instruments=args.instruments)
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

if "test_reconstruct" in args.experiments:
    print("<============test_reconstruct==================>")
    l,predict = disentangle(model, z,p,z_prime,p_prime)

    z_prime_codes = z_prime
    z_prime = model.quantizer.from_codes(z_prime_codes[0].unsqueeze(0))[0]

    out_codes = predict["z"]
    out = model.quantizer.from_codes(out_codes[0].unsqueeze(0))[0]

    with torch.no_grad():
        input_audio = model.decode(z_prime)
        output_audio = model.decode(out)

    
    writer.add_audio(f"Audio/Ground Truth:"  , input_audio[0])
    writer.add_audio(f"Audio/Reconstruction" , output_audio[0])

    writer.flush()

    exit()

if "test_pitch_sweep" in args.experiments:
    print("<============test_pitch_sweep==================>")

    z_codes = z
    z = model.quantizer.from_codes(z_codes[0].unsqueeze(0))[0]

    with torch.no_grad():
        input_audio = model.decode(z)

    
    writer.add_audio(f"Audio/Ground Truth={p[0].item()}:"  , input_audio[0])

    for p_p in torch.tensor([num for num in [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,45,46,47,
    48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,
    75,76,77,78,79,80,81,82,83,84,85,86,87,88,90]]).to(device):
        l,predict = disentangle(model, z_codes[0].unsqueeze(0),p[0].unsqueeze(0),z_prime[0].unsqueeze(0),p_p.unsqueeze(0))
        out_codes = predict["z"]
        out = model.quantizer.from_codes(out_codes[0].unsqueeze(0))[0]

        with torch.no_grad():
            output_audio = model.decode(out)
            
        writer.add_audio(f"Audio/Reconstruction pitch={p_p.item()}" , output_audio[0])

    writer.flush()

    exit()

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

if "autoencoder" in args.experiments:
    print("<============autoencoder==================>")

    _,a_pred = disentangle.train_autoencoder(z_codes)    
    a_z = model.quantizer.from_codes(z_codes[0].unsqueeze(0))[0]

    a_out_codes = torch.argmax(a_pred["z"], dim=1)
    a_out = model.quantizer.from_codes(a_out_codes[0].unsqueeze(0))[0]

    with torch.no_grad():
        a_input_audio = model.decode(a_z)
        a_output_audio = model.decode(a_out)

    writer.add_audio(f"Audio/Autoencoder Ground Truth:"  , a_input_audio[0])
    writer.add_audio(f"Audio/Autoencoder Reconstruction" , a_output_audio[0])

if "reconstruct" in args.experiments:
    print("<============reconstruct==================>")
    writer.add_audio(f"Audio/Ground Truth:"  , input_audio[0])
    writer.add_audio(f"Audio/Reconstruction" , output_audio[0])

    samples = input_audio[0][0]
    mfcc_prime =  mfcc_prime * (data.mfcc_max - data.mfcc_min) + data.mfcc_min
    mfcc_hat = (predict["mfcc"] * (data.mfcc_max - data.mfcc_min) + data.mfcc_min)


    writer.add_image(f"Pitch/Ground Truth", make_pitch_img(samples, p_prime[0]), dataformats='HWC')
    writer.add_image(f"Pitch/Reconstruction", make_pitch_img(samples, p_hat[0]), dataformats='HWC')

    writer.add_image(f"MFCC/Ground Truth", make_mfcc_img(mfcc_prime), dataformats='HWC')
    writer.add_image(f"MFCC/Reconstruction", make_mfcc_img(mfcc_hat), dataformats='HWC')

    writer.add_image(f"RMS/Ground Truth", make_rms_img(rms_prime), dataformats='HWC')
    writer.add_image(f"RMS/Reconstruction", make_rms_img(predict["rms"]), dataformats='HWC')

    z_hat = predict["z"]

if "pitch_sweep" in args.experiments:
    print("<============pitch_sweep==================>")
    input_audio = model.decode(z[0].unsqueeze(0))
    writer.add_audio(f"Audio/Input pitch: {p_old[0]}:"  , input_audio[0])

    pitches = torch.tensor([num for num in [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,45,46,47,
    48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,
    75,76,77,78,79,80,81,82,83,84,85,86,87,88,90]]).to(device)

    for i in range(len([34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,45,46,47,
    48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,
    75,76,77,78,79,80,81,82,83,84,85,86,87,88,90])):
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

if "a_loss" in args.experiments:
    print("<============a_loss==================>")

    losses = []
    for s_z in z:
        with torch.no_grad():
            l,_ = disentangle.train_autoencoder(s_z.unsqueeze(0))  
            losses.append(l["t_predict"].item())

    # print(losses)
    # print(len(losses))

    i_to_family = {128: 1, 642: 3, 387: 7, 644: 5, 65: 4, 8: 6, 905: 4, 656: 0, 914: 0, 150: 0, 921: 6, 414: 1, 927: 0, 417: 0, 675: 8, 420: 0, 37: 10, 40: 4, 43: 1, 46: 10, 50: 4, 436: 8, 183: 7, 440: 6, 572: 1, 701: 6, 958: 6, 577: 4, 450: 8, 488: 5, 838: 4, 327: 4, 457: 3, 590: 5, 82: 2, 803: 0, 609: 8, 86: 2, 219: 3, 805: 4, 224: 7, 97: 8, 100: 8, 263: 3, 872: 0, 316: 3, 880: 0, 104: 7, 759: 0, 121: 1, 378: 3, 123: 6, 510: 3}
    family_to_loss = {0 : [], 1 : [], 2 : [], 3 : [], 4 : [], 5 : [], 6 : [], 7 : [], 8 : [],9 : [], 10 : []}
    
    for i,loss in enumerate(losses):

        #append appropriate family array with the pitch difference calculated
        family_to_loss[i_to_family[inst[i].item()]].append(loss)

    losses = [ np.mean(np.mean(family_to_loss[family])) for family in list(family_to_loss)]

    fig, ax = plt.subplots()

    bar_labels = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']
    bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange', 'aqua', 'pink', 'navy', 'indigo', 'violet', 'yellow', 'cadetblue']

    ax.bar(bar_labels, losses, label=bar_labels, color=bar_colors)

    for label in ax.get_xticklabels():
        label.set_rotation(45)

    ax.set(title=f'Autoencoder reconstruction loss of different instrument families')
    ax.set_xlabel("Instrument Family")
    ax.set_ylabel("NLL Loss value")


    fig.canvas.draw()
    writer.add_image(f"Loss/Loss Instrument Family", grab_buffer(fig), dataformats='HWC')


if "rest_tsne" in args.experiments:
    print("<============rest_tsne==================>") 
    disentangle(z,p,mfcc,rms,inst,z_prime, p_prime, mfcc_prime, rms_prime, inst_prime)

    rest_emb = disentangle.rest_emb[:,:,8]
    d_redux = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=40, n_iter=5000).fit_transform(rest_emb.cpu().detach().numpy())
    kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(d_redux)


    fig, ax = plt.subplots()
    cdict = {759: 'red', 417: 'blue', 644: 'green', 97: 'yellow'}
    i_label = {759: 'bass_electronic_018', 417: 'bass_synthetic_033', 644: 'mallet_acoustic_062', 97: 'string_acoustic_012'}

    for i in np.unique(inst.cpu().detach().numpy()):
        ix = np.where(inst.cpu().detach().numpy() == i)
        ax.scatter(d_redux.T[0][ix], d_redux.T[1][ix], label=i_label[i], color=cdict[i])

    for c in kmeans.cluster_centers_:
        ax.scatter(c.T[0], c.T[1])

    ax.set(title='Rest Embedding TSNE scatter plot')
    ax.legend()
    fig.canvas.draw()
    writer.add_image(f"Rest/TSNE", grab_buffer(fig), dataformats='HWC')
    fig.savefig("test.png")

if "sapd-family" in args.experiments:
    print("<============sapd-family==================>") 
    ip_to_pd = {}

    # Test pitch accuracy for every instrument
    for (z_codes,p_old, mfcc_old, rms_old, inst_old) in zip(z,p,mfcc,rms, inst):
        
        # num += 1
        # if num > 5:
        #     break

        midi_notes = range(21,109)
        random_pitches = random.sample(midi_notes, 5)
        abs_diffs = []

        for i in random_pitches:
            z_predict = disentangle.get_new_sample(z_codes.unsqueeze(0), torch.tensor(i).unsqueeze(0).to(device), p_start=p_old.unsqueeze(0), mfcc=mfcc_old.unsqueeze(0), rms=rms_old.unsqueeze(0), inst=inst_old.unsqueeze(0))
            z_predict = model.quantizer.from_codes(z_predict)[0]
            out = model.decode(z_predict)

            pi = torch.tensor([i]).unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float().to(device)
            pi[:,:, ((z.shape[-1]*disentangle.pc_num)//disentangle.pc_denom):] = torch.tensor(0).to(disentangle.device)

            p_predict = disentangle.pitch_predict(z_predict)
            p_predict = torch.argmax(p_predict, dim=1).unsqueeze(1)
            abs_diff = torch.mean(torch.abs(pi - p_predict)).cpu().detach().numpy().item()
            abs_diffs.append(abs_diff)

            # writer.add_audio(f"Audio/Output-{i}:"  , out[0])
            # writer.add_audio(f"Audio/Predict: pitch = {pitches[i]}" , out[0])

        
        ip_to_pd[str(inst_old.item()) + "_" + str(p_old.item())] = np.mean(abs_diff)
        print(str(inst_old.item()) + "_" + str(p_old.item()))

        # print(ip_to_pd[inst_old.item()])

    i_to_family = {128: 1, 642: 3, 387: 7, 644: 5, 65: 4, 8: 6, 905: 4, 656: 0, 914: 0, 150: 0, 921: 6, 414: 1, 927: 0, 417: 0, 675: 8, 420: 0, 37: 10, 40: 4, 43: 1, 46: 10, 50: 4, 436: 8, 183: 7, 440: 6, 572: 1, 701: 6, 958: 6, 577: 4, 450: 8, 488: 5, 838: 4, 327: 4, 457: 3, 590: 5, 82: 2, 803: 0, 609: 8, 86: 2, 219: 3, 805: 4, 224: 7, 97: 8, 100: 8, 263: 3, 872: 0, 316: 3, 880: 0, 104: 7, 759: 0, 121: 1, 378: 3, 123: 6, 510: 3}
    # i_to_family = {39: 3, 881: 0, 979: 6, 346: 3, 188: 6, 877: 4, 241: 0, 599: 4, 690: 4, 919: 4, 202: 10, 95: 10, 13: 6, 791: 4, 561: 3, 950: 6, 866: 4, 52: 8, 443: 0, 427: 7, 744: 2, 17: 5, 41: 0, 48: 3, 907: 0, 294: 0, 623: 8, 580: 8, 903: 4, 616: 7, 781: 0, 692: 0, 547: 4, 707: 0, 15: 0, 396: 5, 152: 1, 89: 4, 158: 0, 476: 0, 714: 6, 38: 3, 197: 3, 848: 0, 856: 4, 965: 4, 326: 2, 753: 0, 884: 4, 850: 6, 447: 0, 810: 4, 920: 4, 730: 0, 391: 6, 309: 1, 470: 4, 333: 4, 134: 0, 552: 0, 411: 6, 407: 8, 843: 4, 293: 6, 2: 5, 344: 6, 817: 0, 454: 0, 296: 8, 559: 3, 155: 2, 605: 4, 88: 1, 222: 3, 302: 4, 124: 5, 615: 0, 465: 8, 9: 1, 567: 9, 429: 8, 687: 5, 428: 3, 724: 4, 873: 0, 628: 5, 341: 5, 703: 6, 775: 7, 608: 2, 1: 6, 129: 5, 271: 8, 501: 4, 968: 0, 16: 1, 115: 4, 49: 5, 923: 4, 528: 4, 809: 4, 151: 5, 322: 0, 833: 0, 332: 7, 633: 4, 761: 0, 908: 4, 214: 2, 821: 9, 900: 4, 360: 0, 31: 0, 413: 3, 935: 0, 482: 0, 306: 2, 643: 4, 243: 10, 127: 3, 534: 4, 672: 0, 462: 8, 497: 5, 853: 4, 951: 0, 554: 0, 321: 0, 697: 4, 227: 7, 171: 7, 496: 5, 468: 4, 267: 3, 940: 4, 410: 7, 929: 0, 486: 4, 611: 0, 365: 1, 334: 3, 489: 8, 490: 7, 867: 3, 433: 0, 138: 4, 237: 4, 578: 0, 734: 4, 684: 5, 125: 3, 607: 2, 772: 9, 563: 1, 137: 0, 625: 6, 558: 6, 639: 0, 663: 1, 216: 5, 557: 10, 653: 6, 79: 6, 909: 0, 793: 7, 673: 4, 737: 4, 592: 4, 768: 7, 503: 5, 668: 5, 800: 6, 726: 0, 849: 4, 453: 0, 891: 6, 531: 3, 445: 5, 666: 4, 288: 4, 953: 0, 196: 7, 303: 3, 602: 5, 818: 0, 193: 1, 437: 1, 963: 3, 29: 7, 584: 3, 165: 8, 852: 6, 366: 4, 770: 0, 324: 3, 240: 4, 841: 4, 862: 0, 386: 10, 977: 4, 898: 0, 401: 1, 455: 4, 573: 6, 472: 8, 0: 3, 731: 4, 373: 5, 527: 0, 676: 5, 94: 8, 68: 0, 991: 4, 830: 6, 878: 4, 305: 3, 708: 0, 987: 0, 442: 2, 254: 3, 92: 5, 286: 0, 475: 4, 874: 0, 270: 3, 949: 0, 487: 8, 722: 0, 56: 10, 421: 6, 359: 5, 787: 6, 75: 4, 218: 4, 933: 0, 955: 0, 762: 4, 444: 6, 146: 5, 12: 3, 184: 5, 213: 6, 739: 6, 957: 4, 422: 3, 629: 8, 936: 0, 238: 8, 598: 4, 520: 4, 223: 9, 831: 4, 944: 4, 869: 0, 946: 6, 76: 3, 364: 0, 131: 5, 14: 0, 22: 7, 750: 7, 556: 7, 839: 4, 648: 4, 298: 5, 751: 0, 26: 3, 106: 8, 747: 6, 329: 6, 394: 3, 323: 5, 709: 0, 493: 3, 728: 0, 655: 3, 825: 6, 484: 3, 119: 10, 667: 7, 962: 0, 896: 0, 738: 0, 740: 3, 939: 0, 368: 3, 350: 4, 539: 8, 259: 0, 210: 5, 553: 6, 634: 9, 276: 0, 745: 0, 72: 3, 21: 6, 915: 6, 185: 3, 660: 4, 376: 7, 283: 4, 735: 8, 451: 4, 10: 4, 20: 10, 245: 1, 669: 4, 998: 4, 630: 4, 506: 6, 897: 4, 164: 3, 777: 5, 47: 3, 144: 4, 649: 8, 564: 3, 586: 7, 778: 6, 857: 4, 485: 1, 461: 1, 691: 0, 404: 4, 610: 0, 24: 7, 732: 0, 594: 5, 959: 0, 637: 5, 62: 5, 179: 7, 507: 5, 654: 0, 212: 5, 882: 0, 398: 10, 683: 6, 99: 1, 203: 2, 861: 0, 388: 3, 320: 6, 54: 10, 311: 1, 876: 0, 342: 0, 719: 6, 44: 7, 678: 0, 835: 4, 257: 7, 73: 4, 483: 4, 755: 6, 500: 3, 583: 0, 706: 3, 752: 5, 400: 0, 32: 7, 521: 0, 239: 5, 108: 5, 101: 4, 459: 7, 713: 5, 811: 7, 733: 4, 789: 6, 710: 5, 77: 3, 943: 4, 541: 6, 860: 0, 66: 2, 139: 2, 456: 4, 613: 3, 548: 6, 885: 4, 310: 5, 492: 4, 895: 4, 98: 8, 133: 3, 824: 4, 481: 0, 209: 8, 854: 4, 651: 4, 132: 2, 864: 6, 406: 1, 827: 6, 892: 4, 361: 2, 518: 4, 383: 6, 794: 5, 116: 5, 671: 9, 143: 8, 313: 5, 318: 10, 889: 6, 961: 0, 177: 7, 225: 5, 249: 0, 851: 4, 409: 10, 842: 4, 408: 0, 136: 8, 509: 3, 650: 2, 792: 4, 945: 0, 354: 0, 807: 0, 353: 6, 624: 0, 466: 10, 248: 10, 449: 3, 575: 6, 727: 6, 285: 2, 515: 4, 711: 6, 826: 0, 741: 4, 764: 9, 418: 9, 632: 9, 474: 0, 264: 4, 331: 5, 565: 0, 600: 3, 913: 6, 804: 6, 785: 0, 679: 3, 307: 5, 776: 6, 371: 7, 157: 10, 374: 7, 674: 0, 61: 3, 795: 7, 110: 1, 612: 1, 226: 10, 290: 1, 69: 5, 194: 4, 441: 1, 820: 5, 855: 6, 191: 3, 59: 0, 513: 3, 569: 3, 233: 4, 140: 4, 845: 6, 829: 4, 336: 8, 890: 0, 536: 2, 589: 0, 351: 3, 695: 6, 508: 3, 604: 0, 33: 1, 657: 6, 250: 5, 621: 3, 847: 0, 816: 4, 187: 0, 808: 6, 434: 0, 126: 5, 784: 6, 114: 10, 19: 3, 162: 2, 328: 2, 57: 1, 377: 1, 756: 6, 941: 6, 166: 8, 786: 5, 425: 0, 749: 4, 742: 9, 930: 6, 67: 5, 495: 5, 170: 1, 925: 0, 479: 5, 641: 3, 103: 5, 11: 8, 571: 0, 234: 3, 910: 0, 244: 9, 517: 5, 702: 4, 435: 8, 393: 1, 917: 7, 389: 5, 141: 2, 971: 6, 844: 0, 375: 5, 317: 0, 284: 8, 588: 1, 7: 1, 367: 7, 363: 2, 246: 8, 349: 8, 754: 8, 868: 0, 112: 10, 718: 4, 480: 0, 614: 0, 431: 0, 314: 3, 766: 4, 658: 3, 161: 6, 773: 0, 716: 1, 402: 5, 618: 8, 295: 10, 924: 0, 875: 6, 978: 6, 840: 8, 80: 3, 989: 4, 163: 8, 954: 6, 147: 2, 779: 6, 160: 0, 746: 5, 582: 6, 871: 4, 335: 0, 937: 6, 135: 6, 42: 8, 424: 6, 111: 0, 934: 0, 205: 3, 498: 5, 879: 0, 574: 1, 176: 3, 960: 6, 347: 5, 275: 2, 315: 10, 682: 4, 35: 5, 299: 2, 662: 4, 120: 8, 705: 6, 358: 8, 736: 4, 976: 0, 813: 0, 385: 4, 343: 3, 199: 4, 499: 8, 617: 8, 780: 8, 576: 7, 28: 8, 260: 5, 720: 4, 416: 5, 540: 0, 272: 10, 593: 3, 523: 6, 545: 3, 774: 6, 287: 3, 504: 5, 175: 7, 532: 8, 931: 6, 117: 10, 405: 3, 819: 4, 988: 0, 208: 3, 550: 3, 438: 0, 801: 4, 549: 6, 45: 10, 58: 7, 635: 3, 846: 0, 758: 3, 595: 6, 525: 5, 806: 6, 355: 8, 90: 8, 189: 1, 681: 5, 200: 7, 269: 5, 279: 7, 836: 0, 356: 5, 36: 10, 967: 7, 524: 3, 865: 6, 948: 4, 91: 3, 904: 0, 526: 6, 665: 0, 1002: 6, 596: 0, 659: 0, 4: 10, 266: 2, 790: 0, 769: 0, 156: 2, 182: 8, 423: 1, 505: 3, 587: 8, 603: 6, 798: 0, 870: 0, 262: 8, 278: 8, 689: 3, 627: 0, 390: 0, 268: 1, 686: 9, 247: 6, 694: 4, 319: 8, 228: 5, 430: 5, 265: 7, 893: 0, 670: 0, 771: 0, 626: 6, 916: 10, 544: 5, 261: 8, 640: 6, 85: 3, 301: 7, 966: 6, 304: 5, 3: 7, 700: 0, 308: 6, 55: 6, 597: 0, 606: 4, 397: 4, 530: 0, 297: 6, 932: 4, 601: 4, 181: 0, 74: 3, 511: 4, 585: 0, 432: 7, 638: 4, 168: 1, 717: 6, 704: 0, 87: 8, 102: 1, 201: 4, 748: 8, 729: 8, 34: 1, 560: 2, 178: 1, 230: 1, 381: 0, 685: 1, 5: 0, 18: 5, 974: 0, 83: 2, 71: 3, 370: 4, 783: 8, 255: 6, 300: 10, 277: 8, 477: 5, 229: 0, 252: 8, 698: 4, 395: 2, 148: 8, 542: 7, 282: 1, 251: 3, 832: 0, 828: 7, 562: 2, 782: 6, 469: 5, 799: 4, 863: 8, 802: 9, 763: 4, 715: 4, 403: 8, 947: 0, 186: 8, 172: 4, 51: 8, 113: 7, 192: 5, 463: 7, 145: 1, 478: 0, 154: 1, 760: 6, 535: 3, 823: 0, 25: 5, 221: 1, 207: 7, 622: 6, 473: 6, 699: 8, 902: 6, 512: 0, 519: 0, 352: 8, 723: 0, 579: 0, 886: 5, 118: 8, 382: 5, 980: 6, 312: 8, 984: 0, 415: 5, 439: 7, 280: 6, 220: 0, 235: 8, 215: 1, 217: 7, 348: 10, 859: 6, 23: 8, 858: 0, 712: 1, 107: 1, 70: 0, 918: 6, 494: 1, 964: 6, 815: 0, 281: 10, 448: 1, 30: 7, 757: 2, 765: 6, 458: 5, 636: 8, 446: 5, 901: 0, 996: 6, 538: 3, 467: 10, 27: 8, 664: 0, 105: 5, 379: 8, 788: 5, 357: 10, 568: 1, 206: 10, 338: 2, 543: 6, 620: 6, 837: 4, 96: 10, 426: 8, 471: 8, 369: 10, 491: 4, 970: 0, 743: 0, 419: 1, 693: 3, 696: 4, 652: 8, 180: 5, 822: 6, 591: 0, 680: 8, 814: 5, 63: 8, 6: 1, 291: 7, 797: 5, 529: 0, 516: 1, 198: 10, 60: 8, 289: 1, 812: 0, 211: 1, 153: 1, 911: 0, 994: 5, 242: 2, 130: 10, 581: 0, 195: 5, 372: 10, 555: 2, 81: 7, 93: 7, 340: 4, 834: 4, 677: 6, 231: 3, 796: 7, 258: 8, 337: 5, 661: 2, 952: 3, 412: 7, 912: 6, 570: 4, 997: 8, 109: 1, 380: 8, 384: 8, 1004: 6, 122: 10, 345: 3, 894: 0, 972: 6, 169: 6, 330: 2, 551: 8, 899: 6, 142: 7, 537: 6, 546: 0, 999: 6, 983: 1, 362: 10, 204: 1, 339: 2, 273: 7, 566: 10, 992: 0, 159: 1, 721: 5, 502: 8, 1001: 6, 256: 7, 190: 8, 464: 3, 53: 10, 78: 7, 725: 6, 533: 6, 325: 10, 84: 6, 522: 7, 514: 0, 619: 8, 399: 7, 986: 6, 926: 4, 985: 6, 174: 10, 292: 8, 392: 5, 973: 6, 982: 6, 969: 0, 928: 6, 64: 1, 767: 2, 460: 1, 887: 6, 173: 1, 631: 5, 888: 0, 995: 0, 922: 2, 452: 3, 274: 1, 149: 10, 1003: 2, 993: 6, 981: 4, 236: 7, 883: 2, 1005: 0, 938: 2, 906: 0, 956: 10, 647: 3, 253: 10, 645: 5, 990: 8, 646: 6, 232: 2, 167: 10, 688: 2, 942: 10, 975: 5, 1000: 1}
    family_to_pd = {0 : [], 1 : [], 2 : [], 3 : [], 4 : [], 5 : [], 6 : [], 7 : [], 8 : [],9 : [], 10 : []}

    for ip in list(ip_to_pd):

        #append appropriate family array with the pitch difference calculated
        family_to_pd[i_to_family[int(ip.split("_")[0])]].append(int(ip.split("_")[1]))


    pds = [ np.mean(np.mean(family_to_pd[family]))for family in list(family_to_pd)]

    fig, ax = plt.subplots()

    bar_labels = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']
    bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange', 'aqua', 'pink', 'navy', 'indigo', 'violet', 'royalblue', 'cadetblue']

    ax.bar(bar_labels, pds, label=bar_labels, color=bar_colors)

    ax.set(title=f'Stochastic Absolute pitch of different instrument families')
    ax.set_xlabel("Instrument Family")
    ax.set_ylabel("Absolute Pitch Difference")


    fig.canvas.draw()
    writer.add_image(f"Pitch/Pitch Differences Per Instrument Family", grab_buffer(fig), dataformats='HWC')

if "sapd-note" in args.experiments:
    print("<============sapd-note==================>") 

    # Test pitch accuracy for every instrument
    midi_notes = range(21,109)  
    note_to_pd = { note: [] for note in midi_notes}  

    for current_note in midi_notes:
        print(current_note)

        #get random 300 subset from test set against every notte
        for (z_codes,p_old, mfcc_old, rms_old, inst_old) in random.sample(list(zip(z,p,mfcc,rms, inst)),300):

            

            abs_diffs = []

            z_predict = disentangle.get_new_sample(z_codes.unsqueeze(0), torch.tensor(current_note).unsqueeze(0).to(device), p_start=p_old.unsqueeze(0), mfcc=mfcc_old.unsqueeze(0), rms=rms_old.unsqueeze(0), inst=inst_old.unsqueeze(0))
            z_predict = model.quantizer.from_codes(z_predict)[0]
            out = model.decode(z_predict)

            pi = torch.tensor([current_note]).unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float().to(device)
            pi[:,:, ((z.shape[-1]*disentangle.pc_num)//disentangle.pc_denom):] = torch.tensor(0).to(disentangle.device)

            p_predict = disentangle.pitch_predict(z_predict)
            p_predict = torch.argmax(p_predict, dim=1).unsqueeze(1)
            abs_diff = torch.mean(torch.abs(pi - p_predict)).cpu().detach().numpy().item()

                # writer.add_audio(f"Audio/Output-{i}:"  , out[0])
                # writer.add_audio(f"Audio/Predict: pitch = {pitches[i]}" , out[0])

            
            note_to_pd[current_note].append(abs_diff)
    

            # print(ip_to_pd[inst_old.item()])



    pds = [ np.mean(np.mean(note_to_pd[note])) for note in midi_notes]

    fig, ax = plt.subplots()

    bar_labels = [ str(note) for note in midi_notes]
    # bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange', 'aqua', 'pink', 'navy', 'indigo', 'violet', 'royalblue', 'cadetblue']

    ax.bar(bar_labels, pds, label=bar_labels, )

    ax.set(title=f'Stochastic Absolute pitch of different transformed notes families')
    ax.set_xlabel("MIDI Note", fontsize=6)
    ax.set_ylabel("Absolute Pitch Difference")


    fig.canvas.draw()
    writer.add_image(f"Pitch/Pitch Differences Per Transformed MIDI Note", grab_buffer(fig), dataformats='HWC')


cdict = {759: 'red', 417: 'blue', 644: 'green', 97: 'yellow'}
i_label = {759: 'bass_electronic_018', 417: 'bass_synthetic_033', 644: 'mallet_acoustic_062', 97: 'string_acoustic_012'}

if "pd_matrix" in args.experiments:
    pd_matrixes = {}
    print("<============pd_matrix==================>") 
    for (z_codes,p_old, mfcc_old, rms_old, inst_old) in zip(z,p,mfcc,rms, inst):
        
        # num += 1
        # if num > 5:
        #     break

        midi_notes = range(21,109)
        abs_diffs = []

        for i in midi_notes:
            z_predict = disentangle.get_new_sample(z_codes.unsqueeze(0), torch.tensor(i).unsqueeze(0).to(device), p_start=p_old.unsqueeze(0), mfcc=mfcc_old.unsqueeze(0), rms=rms_old.unsqueeze(0), inst=inst_old.unsqueeze(0))
            z_predict = model.quantizer.from_codes(z_predict)[0]
            out = model.decode(z_predict)

            pi = torch.tensor([i]).unsqueeze(1).expand(-1, z.shape[-1]).unsqueeze(1).float().to(device)
            pi[:,:, ((z.shape[-1]*disentangle.pc_num)//disentangle.pc_denom):] = torch.tensor(0).to(disentangle.device)

            p_predict = disentangle.pitch_predict(z_predict)
            p_predict = torch.argmax(p_predict, dim=1).unsqueeze(1)
            abs_diff = torch.mean(torch.abs(pi - p_predict)).cpu().detach().numpy().item()
            abs_diffs.append(abs_diff)

            # writer.add_audio(f"Audio/Output-{i}:"  , out[0])
            # writer.add_audio(f"Audio/Predict: pitch = {pitches[i]}" , out[0])
        if inst_old.item() in pd_matrixes:
            pd_matrixes[inst_old.item()].update({p_old.item(): abs_diffs})

        else:
            pd_matrixes[inst_old.item()] = {p_old.item(): abs_diffs}
        print(pd_matrixes[inst_old.item()])


    for i_num in list(pd_matrixes):

        matrix = []
        s_notes = sorted(pd_matrixes[i_num])

        for s_note in s_notes:
            matrix.append(pd_matrixes[i_num][s_note])

        matrix = np.array(matrix)

        fig, ax = plt.subplots()

        caxes = ax.matshow(matrix)
        cbar = fig.colorbar(caxes)
        cbar.ax.set_ylabel('Abs Difference', rotation=270)

        ax.set(title=f'Absolute pitch differences of {i_label[i_num]}')
        ax.set_xticks(np.arange(len(range(21,109))))
        ax.set_xticklabels(list(range(21,109)))
        ax.set_yticks(np.arange(len(s_notes)))
        ax.set_yticklabels(s_notes)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
        ax.set_xlabel("trabsformed midinote")
        ax.set_ylabel("input midinote")


        fig.canvas.draw()
        writer.add_image(f"Pitch/Sweep for inst {i_num}", grab_buffer(fig), dataformats='HWC')

        fig, ax = plt.subplots()

        ax.plot(list(midi_notes), abs_diffs)
        ax.set(title=f'Absolute difference of predict pitches (against pitch {p_old})')
        ax.set_xlabel("Midi note")
        ax.set_ylabel("Abs Difference")

        fig.canvas.draw()
        writer.add_image(f"Pitch/Instrument {inst_old} Pitch {p_old} Sweep ", grab_buffer(fig), dataformats='HWC')
