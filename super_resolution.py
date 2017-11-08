import argparse

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor

# Test settings
parser = argparse.ArgumentParser(description='Test Super Resolution')
parser.add_argument('--image_name', type=str, required=True, help='input image to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
opt = parser.parse_args()

img = Image.open('images/' + opt.image_name).convert('YCbCr')
y, cb, cr = img.split()

model = torch.load('checkpoints/' + opt.model)
image = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])

if torch.cuda.is_available():
    model = model.cuda()
    image = image.cuda()

out = model(image)
out = out.cpu()
out_img_y = out.data[0].numpy()
out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)
out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

out_img.save('results/' + opt.image_name)
print('output image saved to ', 'results/' + opt.image_name)
