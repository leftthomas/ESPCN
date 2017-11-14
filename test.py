import os
from os import listdir

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from tqdm import tqdm

from data_utils import is_image_file
from model import Net

UPSCALE_FACTOR = 3
MODEL_NAME = 'epoch_200.pt'

if __name__ == "__main__":
    path = 'data/test/SRF_' + str(UPSCALE_FACTOR) + '/data/'
    images_name = [x for x in listdir(path) if is_image_file(x)]
    model = Net(upscale_factor=UPSCALE_FACTOR)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

    out_path = 'results/SRF_' + str(UPSCALE_FACTOR)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for image_name in tqdm(images_name, desc='convert LR images to HR images'):
        image = Image.open(path + image_name)
        image = Variable(ToTensor()(image))
        if torch.cuda.is_available():
            image = image.cuda()

        out = model(image).cpu().data[0].numpy()
        out_img = Image.fromarray(out)
        out_img.save(out_path + image_name)
