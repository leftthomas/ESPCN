# Super Resolution using an efficient sub-pixel convolutional neural network

The implement of the method described in  ["Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" - Shi et al.](https://arxiv.org/abs/1609.05158) for increasing spatial resolution.

```
usage: main.py [-h] --upscale_factor UPSCALE_FACTOR 

optional arguments:
  -h, --help            show this help message and exit
  --upscale_factor      super resolution upscale factor
```
This super-resolution network trains on the [BSD300 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), using crops from the 200 training images, and evaluating on crops of the 100 test images. A snapshot of the model after every epoch with filename model_epoch_<epoch_number>.pth

## Usage

### Train

`python main.py --upscale_factor 3`

### Test
`python super_resolve.py --input_image 16077.jpg --model model_epoch_5.pth --output_filename out.png`
