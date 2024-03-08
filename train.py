import data
import model
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import yaml
import auraloss


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
sr = training_params["sample_rate"]

#log for tensorboard
writer = SummaryWriter(training_params["tb_path"])


# #get training and validation datasets

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data = data.audio_data(data_path, sr)
train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
# print(next(iter(dataLoader)))

network = model.VQVAE(training_params["in_channels"],
        training_params["num_hidddens"],
        training_params["num_downsampling_layers"],
        training_params["num_residual_layers"],
        training_params["num_residual_hiddens"],
        training_params["embedding_dim"],
        training_params["num_embeddings"],
        training_params["use_ema"],
        training_params["decay"],
        training_params["epsilon"],)
network = network.to(device)


# Initialize optimizer.
train_params = [params for params in network.parameters()]
lr = training_params["lr"]
optimizer = optim.Adam(train_params, lr=lr)
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
network.train()
for epoch in range(training_params["epochs"]):
    total_train_loss = 0
    total_recon_error = 0
    n_train = 0
    for (batch_idx, train_tensors) in enumerate(train_loader):
        optimizer.zero_grad()
        audio = train_tensors[0].to(device)
        out = network(audio)
        recon_error = criterion(out["x_recon"], audio)
        total_recon_error += recon_error.item()
        loss = recon_error + beta * out["commitment_loss"]
        if not training_params["use_ema"]:
            loss += out["dictionary_loss"]

        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        n_train += 1

        if ((batch_idx + 1) % eval_every) == 0:
            print(f"epoch: {epoch}\nbatch_idx: {batch_idx + 1}", flush=True)
            total_train_loss /= n_train
            if total_train_loss < best_train_loss:
                best_train_loss = total_train_loss
                torch.save(network, training_params["save_path"])

            print(f"total_train_loss: {total_train_loss}")
            print(f"best_train_loss: {best_train_loss}")
            print(f"recon_error: {total_recon_error / n_train}\n")

            writer.add_scalar("Loss/Total", total_train_loss, epoch)
            writer.add_scalar("Loss/Reconstruction", total_recon_error, epoch)

            total_train_loss = 0
            total_recon_error = 0
            n_train = 0
    

# Generate and save reconstructions.
network.eval()

audio = next(iter(train_loader))
out = network(audio[0].to(device))

writer.add_audio("Input Audio "  , audio[0][0], 0, 16000)
writer.add_audio("Reconstuction " , out["x_recon"][0], 0, 16000)

writer.flush()
writer.close()

# valid_dataset = CIFAR10(data_root, False, transform, download=True)
# valid_loader = DataLoader(
#     dataset=valid_dataset,
#     batch_size=batch_size,
#     num_workers=workers,
# )

# with torch.no_grad():
#     for valid_tensors in valid_loader:
#         break

#     save_img_tensors_as_grid(valid_tensors[0], 4, "true")
#     save_img_tensors_as_grid(network(valid_tensors[0].to(device))["x_recon"], 4, "recon")

# print(zeros.shape)
# print(data[0].shape)
# validation_data = data.labeled_data(data_path, "validation", data.get_transform(train=True))

# #get training and validation dataloaders
# training_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# # label_criterion = torch.nn.CrossEntropyLoss()
# label_criterion = torch.nn.CrossEntropyLoss()
# bbox_criterion = torch.nn.MSELoss()
# score_criterion = torch.nn.MSELoss()

# # train on the GPU or on the CPU, if a GPU is not available


# #load model
# network = model.VGG(device=device)
# network = network.to(device)

# #initialize optimizer
# optimizer = optim.SGD(network.parameters(), lr=lr, momentum=0.9)

# #log for tensorboard
# writer = SummaryWriter(save_path[:-3] + "_runs")

# #set start epoch
# start_epoch = 0

# #load checkpoint if fle exists
# if os.path.exists(save_path):

#     checkpoint = torch.load(save_path, map_location=device)

#     network.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = checkpoint['epoch']


# for epoch in range(start_epoch, epochs):
#     running_loss = 0.0
#     label_loss = 0 
#     bbox_loss = 0
#     score_loss = 0
    
#     for ind, (inputs,labels) in enumerate(training_loader, 0):
        
#         inputs = [img.to(device) for img in inputs]
#         t_labels = [l['labels'].to(device) for l in labels]
#         t_bboxes = [l['bboxes'].to(device) for l in labels]

#         # zero the parameter gradients
#         optimizer.zero_grad()
#         loss =0
#         # forward + backward + optimize 
#         p_out = network(inputs)
#         for i,p_dict in enumerate(p_out):
#             for j in range(model.num_boxes):
#                 num_boxes = len(t_labels[i])
#                 #calculate loss when ground truth bboxes are available
#                 if j < num_boxes:
#                     label_loss_j = label_criterion(p_dict["labels"][j],t_labels[i][j])
#                     bbox_loss_j = torchvision.ops.distance_box_iou_loss(p_dict["boxes"][j],t_bboxes[i][j])
#                     score_loss_j = score_criterion(p_dict["scores"][j],torch.tensor(1.0).to(device))
#                     label_loss += label_loss_j
#                     bbox_loss += bbox_loss_j
#                     score_loss += score_loss_j
#                     loss += sum([label_loss_j, bbox_loss_j, score_loss_j])
#                 #only calculate loss on score when there is not ground truth bbox
#                 else:
#                     score_loss_j = score_criterion(p_dict["scores"][j],torch.tensor(0.0).to(device))
#                     score_loss += score_loss_j
#                     loss +=  score_loss_j


#         loss.backward()
#         optimizer.step()
        
#         # print statistics
#         running_loss += loss.item()
#         if ind % 100 == 99:    # print every 100 mini-batches
#             print(f'[{epoch + 1}, {ind + 1:5d}] loss: {running_loss / 100:.3f}')
#             running_loss = 0.0

#     #save model every 10 epochs
#     if epoch % 10 == 0:
#         torch.save({'epoch': epoch,
#             'model_state_dict': network.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss}, save_path)
    
#     #document loss of epoch
#     writer.add_scalar("Loss/label", label_loss, epoch)
#     writer.add_scalar("Loss/bboxes", bbox_loss, epoch)
#     writer.add_scalar("Loss/score", score_loss, epoch)      
#     writer.add_scalar("Loss/all", label_loss+bbox_loss+score_loss, epoch)    

# #write image results to tensor board
# network.train(False)

# images, labels = next(iter(validation_loader))
# inputs = [img.to(device) for img in images]
# p_boxes = [dic["boxes"] for dic in network(inputs)]
# p_labels = [dic["labels"] for dic in network(inputs)]

# #make sure x_min < x_max and y_min < y_max
# for i in range(len(p_boxes)):
#     for j in range(len(p_boxes[i])):
#         if p_boxes[i][j][0] > p_boxes[i][j][2]:
#             p_boxes[i][j][2] = p_boxes[i][j][0] +1
#         if p_boxes[i][j][1] > p_boxes[i][j][3]:
#             p_boxes[i][j][3] = p_boxes[i][j][1] +1



# truth_images = [torchvision.utils.draw_bounding_boxes(image, labels[i]["bboxes"], labels=[list(data.class_dict.keys())[int(l)] for l in labels[i]["labels_i"]] ) for i,image in enumerate(images)]
# predict_images = [torchvision.utils.draw_bounding_boxes(image, p_boxes[i], labels=[list(data.class_dict.keys())[int(l)] for l in p_labels[i]] ) for i,image in enumerate(images)]

# #display predicted images on tesorboard
# for i in range(len(images)):
#     writer.add_image("truth image: "  +str(i), truth_images[i])
#     writer.add_image("predict image: "  +str(i), predict_images[i])

# #close tensorboard writer
# writer.flush()
# writer.close()

# #save model
# torch.save({'epoch': epochs,
#             'model_state_dict': network.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': 0}, save_path)
   