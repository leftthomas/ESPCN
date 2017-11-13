# Super Resolution using an efficient sub-pixel convolutional neural network

The implement of the method described in  ["Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" - Shi et al.](https://arxiv.org/abs/1609.05158) for increasing spatial resolution.

This super-resolution network trains on the [BSD300 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), using crops from the 200 training images, and evaluating on crops of the 100 test images. A snapshot of the model after every epoch with filename model_epoch_<epoch_number>.pth

## Usage

### Train

`python main.py`

### Test
`python super_resolution.py --image_name 299086.jpg --model model_epoch_100.pth`
