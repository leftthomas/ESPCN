from os import listdir

from PIL import Image
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale
from tqdm import tqdm


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Scale(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def generate_dataset(data_type, upscale_factor):
    images_name = [x for x in listdir('data/VOC2012/' + data_type) if is_image_file(x)]
    crop_size = calculate_valid_crop_size(256, upscale_factor)
    lr_transform = input_transform(crop_size, upscale_factor)
    hr_transform = target_transform(crop_size)

    for image_name in tqdm(images_name, desc='generate ' + data_type + ' dataset with upscale factor = '
            + upscale_factor + ' from VOC2012'):
        image = Image.open('data/VOC2012/' + data_type + '/' + image_name)
        target = image.copy()
        image = lr_transform(image)
        target = hr_transform(target)
        image.save('data/' + data_type + '/' + 'SRF_' + upscale_factor + '/' + 'data/' + image_name)
        target.save('data/' + data_type + '/' + 'SRF_' + upscale_factor + '/' + 'target/' + image_name)


if __name__ == "__main__":
    generate_dataset(data_type='train', upscale_factor=2)
    generate_dataset(data_type='val', upscale_factor=2)
    generate_dataset(data_type='train', upscale_factor=3)
    generate_dataset(data_type='val', upscale_factor=3)
    generate_dataset(data_type='train', upscale_factor=4)
    generate_dataset(data_type='val', upscale_factor=4)
    generate_dataset(data_type='train', upscale_factor=8)
    generate_dataset(data_type='val', upscale_factor=8)
