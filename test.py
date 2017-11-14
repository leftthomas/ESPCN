from os import listdir

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from tqdm import tqdm

from data_utils import is_image_file
from model import Net

images_name = [x for x in listdir('images') if is_image_file(x)]
MODEL_NAME = 'epoch_200.pt'
model = Net(upscale_factor=3)
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

for image_name in tqdm(images_name, desc='convert LR images to SR images'):
    img = Image.open('images/' + image_name).convert('YCbCr')
    y, cb, cr = img.split()
    image = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
    if torch.cuda.is_available():
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

    out_img.save('results/' + image_name)
