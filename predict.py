from __future__ import print_function
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from data import input_transform

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()

print('#', opt)
img = Image.open(opt.input_image)

model = torch.load(opt.model)
input_transform = input_transform()
input = input_transform(img).view(1, -1, img.size[1], img.size[0])

if opt.cuda:
    model = model.cuda()
    input = input.cuda()

out = model(input)
out = out.cpu()
out_img = out[0].detach().numpy()
out_img *= 255.0
out_img = out_img.clip(0, 255)
out_img = np.squeeze(out_img)
out_img = Image.fromarray(np.uint8(out_img))

out_img.save(opt.output_filename)
print('# output image saved to ', opt.output_filename)
