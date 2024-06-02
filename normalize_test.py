import torch
import dataset
import librosa
import argparse

parser = argparse.ArgumentParser(description='Pick which feature to compute')
parser.add_argument('-f', '--feature', help="feature to compute", default="mfcc")
args = parser.parse_args()

feature_index = {"mfcc" : 1, "pitch": 2, "rms": 3}

if args.feature not in feature_index:
    raise Exception("Feature not in list of valid features (mfcc pitch rms)")

data = dataset.NSynth(
        "/vast/df2322/data/Nsynth/nsynth-test")
loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)





max_feature = -float("inf")
min_feature = float("inf")

for i,batch in enumerate(loader):
    print(i)
    samples,feature,pitch,rms = batch

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

