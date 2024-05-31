import dataset
import torch
import argparse
import torch

parser = argparse.ArgumentParser(description='Data Directory')
parser.add_argument('-d', '--data_dir', help="directory of data to save features", default="/nsynth-train")
parser.add_argument('-o', '--output', help="output directory and name", default="train_tensor")
parser.add_argument('-m', '--modulus', help="modulus amount for file split", default=25000)
args = parser.parse_args()

data_dir = args.data_dir
output = args.output
mod = args.modulus

batch_size = 1

data = dataset.NSynth(data_dir)
d_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

output_tensor = []

file_index=0
for (batch_idx, train_tensors) in enumerate(d_loader):
    print(f"{batch_idx+1} out of {len(d_loader)}")
    output_tensor.append(train_tensors)
    if batch_idx != 0:
        if (batch_idx % mod) == 0:
            print(f"FILE CREATED: {file_index}")
            torch.save(output_tensor, output + "_" + str(file_index) + ".pt")
            output_tensor = []
            file_index += 1

torch.save(output_tensor, output + "_" + str(file_index) + ".pt")
print("DONE!!!")
