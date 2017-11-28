import argparse
import os
from os import listdir

import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from tqdm import tqdm

from data_utils import is_video_file
from model import Net

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Super Resolution')
    parser.add_argument('--upscale_factor', default=3, type=int, help='super resolution upscale factor')
    parser.add_argument('--is_real_time', default=False, type=bool, help='super resolution real time to show')
    parser.add_argument('--model_name', default='epoch_3_100.pt', type=str, help='super resolution model name')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    IS_REAL_TIME = opt.is_real_time
    MODEL_NAME = opt.model_name

    path = 'data/test/SRF_' + str(UPSCALE_FACTOR) + '/video/'
    videos_name = [x for x in listdir(path) if is_video_file(x)]
    model = Net(upscale_factor=UPSCALE_FACTOR)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

    out_path = 'results/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for video_name in tqdm(videos_name, desc='convert LR videos to HR videos'):
        videoCapture = cv2.VideoCapture(video_name)
        fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
        size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
                int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
        videoWriter = cv2.VideoWriter(out_path + video_name, cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), fps, size)
        # read frame
        success, frame = videoCapture.read()
        while success:
            cv2.imshow('SR Video', frame)
            cv2.waitKey(1000 / int(fps))
            # save video
            videoWriter.write(frame)
            # next frame
            success, frame = videoCapture.read()

        if IS_REAL_TIME:
            img = Image.open(path + video_name).convert('YCbCr')
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
            out_img.save(out_path + video_name)
