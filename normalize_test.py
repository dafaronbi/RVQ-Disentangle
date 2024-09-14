import torch
import dataset
import librosa
import argparse

parser = argparse.ArgumentParser(description='Pick which feature to compute')
parser.add_argument('-f', '--feature', help="feature to compute", default="mfcc")
args = parser.parse_args()

feature_index = {"z":0,  "pitch": 1, "mfcc" : 2, "rms": 3}
UN_min = {"z": -23.3863525390625, "mfcc" : -968.44970703125, "pitch": 0, "rms": 0}
UN_max = {"z":26.479732513427734, "mfcc" : 341.15484619140625, "pitch": 2093.004522404789, "rms": 0.9348938465118408}

if args.feature not in feature_index:
    raise Exception("Feature not in list of valid features (mfcc pitch rms)")

print("LOADING DATA...")
# data = dataset.NSynth_ram(["train_tensor_" + str(i) + ".pt" for i in range(12)])
# data = dataset.NSynth_ram(["test_tensor.pt"])
# data = dataset.NSynth("/nsynth-train")

d_path = ["train_tensor_JC_" + str(i) + ".pt" for i in range(24)]
data = dataset.NSynth_transform_ram(d_path)
loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
print("DONE!!")


max_feature = -float("inf")
min_feature = float("inf")

for i,batch in enumerate(loader):
    print(i)
    z,feature,pitch,rms, z_prime = batch

    feature = batch[feature_index[args.feature]]

    local_max = torch.max(feature)
    local_min = torch.min(feature)

    print("Current Max " + args.feature + f" is: {max_feature}")
    print("Current Min "+ args.feature + f" is: {min_feature}")    
    print("Local Max "+ args.feature + f" is: {local_max}")
    print("Local Min "+ args.feature + f" is: {local_min}")

    if local_max > max_feature:
        max_feature = local_max
    
    if local_min < min_feature:
        min_feature = local_min

print(f"Max " + args.feature + f" is: {max_feature}")
print(f"Min " + args.feature + f" is: {min_feature}")

