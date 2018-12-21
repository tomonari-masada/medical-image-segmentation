from __future__ import print_function
import argparse
from math import log10

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import FullyConvolutionalNetwork
from data import get_data

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation Example')
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--valBatchSize', type=int, default=10, help='validation batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

print('#', opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)

device = torch.device("cuda" if opt.cuda else "cpu")

print('# Loading datasets')
train_set = get_data('train')
val_set = get_data('val')
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
validation_data_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.valBatchSize, shuffle=False)

print('# Building model')
model = FullyConvolutionalNetwork(out_h=256, out_w=256).to(device)
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)


def train(epoch):
    epoch_loss = 0.
    epoch_iou = 0.
    for iteration, batch in enumerate(training_data_loader, 1):
        input, label = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, label)
        epoch_loss += loss.item() * len(input)
        loss.backward()
        optimizer.step()

        output = torch.sigmoid(output.detach())
        output, label = output.cpu().numpy(), label.cpu().numpy()
        output = np.round(output)
        output, label = output[:, 0, ...], label[:, 0, ...]
        inter = (output * label).sum(axis=(1,2))
        union = (output + label - output * label).sum(axis=(1,2))
        epoch_iou += (inter / union).sum()

    if epoch % 10 == 0:
        print("### Epoch {}: Avg. Loss: {:.6f}: Avg. mIOU: {:.6f}".format(
            epoch, epoch_loss / len(train_set), epoch_iou / len(train_set)))


def validation():
    avg_loss = 0.
    avg_iou = 0.
    with torch.no_grad():
        for batch in validation_data_loader:
            input, label = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            loss = criterion(prediction, label)
            avg_loss += loss * len(input)

            output = torch.sigmoid(prediction)
            output, label = output.cpu().numpy(), label.cpu().numpy()
            output = np.round(output)
            output, label = output[:, 0, ...], label[:, 0, ...]
            inter = (output * label).sum(axis=(1,2))
            union = (output + label - output * label).sum(axis=(1,2))
            avg_iou += (inter / union).sum()
            
    print('# Avg. Loss: {:.6f} : Avg. mIOU: {:.6f}'.format(
        avg_loss / len(val_set), avg_iou / len(val_set)))


def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("# model saved to {}".format(model_out_path))

for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    if epoch % 10 == 0:
        validation()
    if epoch % 50 == 0:
        checkpoint(epoch)
