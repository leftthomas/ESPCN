# Super Resolution
A PyTorch implementation of ESPCN based on CVPR2016 paper 
[Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158)

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch
```
conda install pytorch torchvision -c soumith
conda install pytorch torchvision cuda80 -c soumith # install it if you have installed cuda
```
- PyTorchNet
```
pip install git+https://github.com/pytorch/tnt.git@master
```
- tqdm
```
pip install tqdm
```

## Datasets

### Train、Val Dataset
The train and val datasets are sampled from [VOC2012](http://cvlab.postech.ac.kr/~mooyeol/pascal_voc_2012/).
Train dataset has 16700 images and Val dataset has 425 images.
Download the datasets from [here](https://pan.baidu.com/s/1c17nfeo), 
and then extract it into `data` directory. Finally run
```
python data_utils.py

optional arguments:
--upscale_factor      super resolution upscale factor [default value is 3]
```
to generate train and val datasets from VOC2012 with given upscale factors(options: 2、3、4、8).

### Test Dataset
The test dataset are sampled from 
| **Set 5** |  [Bevilacqua et al. BMVC 2012](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)
| **Set 14** |  [Zeyde et al. LNCS 2010](https://sites.google.com/site/romanzeyde/research-interests)
| **BSD 100** | [Martin et al. ICCV 2001](https://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
| **Sun-Hays 80** | [Sun and Hays ICCP 2012](http://cs.brown.edu/~lbsun/SRproj2012/SR_iccp2012.html)
| **Urban 100** | [Huang et al. CVPR 2015](https://sites.google.com/site/jbhuang0604/publications/struct_sr).
Download the dataset from [here](https://pan.baidu.com/s/1nuGyn8l), and then extract it into `data` directory.

## Usage

### Train

```
python -m visdom.server & python train.py

optional arguments:
--upscale_factor      super resolution upscale factor [default value is 3]
--num_epochs          super resolution epochs number [default value is 100]
```
Visdom now can be accessed by going to `127.0.0.1:8097` in your browser, 
or your own host address if specified.

If the above does not work, try using an SSH tunnel to your server by 
adding the following line to your local `~/.ssh/config` :
`LocalForward 127.0.0.1:8097 127.0.0.1:8097`.

Maybe if you are in China, you should download the static resources from 
[here](https://pan.baidu.com/s/1hr80UbU), and put them on 
`~/anaconda3/lib/python3.6/site-packages/visdom/static/`.

### Test
```
python test.py

optional arguments:
--upscale_factor      super resolution upscale factor [default value is 3]
--model_name          super resolution model name [default value is epoch_3_100.pt]
```
The output high resolution images are on `results` directory.

## Benchmarks
Adam optimizer were used with learning rate scheduling between epoch 30 and epoch 80. 

**Upscale Factor = 3**

Epochs with batch size of 64 takes ~30 seconds on a NVIDIA GeForce GTX TITAN X GPU. 

> Loss/PSNR graphs

<table>
  <tr>
    <td>
     <img src="images/3_trainloss.png"/>
    </td>
    <td>
     <img src="images/3_valloss.png"/>
    </td>
  </tr>
</table>
<table>
  <tr>
    <td>
     <img src="images/3_trainpsnr.png"/>
    </td>
    <td>
     <img src="images/3_valpsnr.png"/>
    </td>
  </tr>
</table>

> Results

The left is low resolution image, the middle is high resolution image, and 
the right is super resolution image(output of the ESPCN).

- Set5
<table>
  <tr>
    <td>
     <img src="images/3_LR_Set5_004.png"/>
    </td>
    <td>
     <img src="images/3_HR_Set5_004.png"/>
    </td>
    <td>
     <img src="images/3_SR_Set5_004.png"/>
    </td>
  </tr>
</table>

- Set14
<table>
  <tr>
    <td>
     <img src="images/3_LR_Set14_001.png"/>
    </td>
    <td>
     <img src="images/3_HR_Set14_001.png"/>
    </td>
    <td>
     <img src="images/3_SR_Set14_001.png"/>
    </td>
  </tr>
</table>

- BSD100
<table>
  <tr>
    <td>
     <img src="images/3_LR_BSD100_063.png"/>
    </td>
    <td>
     <img src="images/3_HR_BSD100_063.png"/>
    </td>
    <td>
     <img src="images/3_SR_BSD100_063.png"/>
    </td>
  </tr>
</table>


**Upscale Factor = 4**

Epochs with batch size of 64 takes ~1 minute on a NVIDIA GeForce GTX TITAN X GPU. 

> Loss/PSNR graphs

<table>
  <tr>
    <td>
     <img src="images/3_trainloss.png"/>
    </td>
    <td>
     <img src="images/3_valloss.png"/>
    </td>
  </tr>
</table>
<table>
  <tr>
    <td>
     <img src="images/3_trainpsnr.png"/>
    </td>
    <td>
     <img src="images/3_valpsnr.png"/>
    </td>
  </tr>
</table>

> Results

The left is low resolution image, the middle is high resolution image, and 
the right is super resolution image(output of the ESPCN).

- Set5
<table>
  <tr>
    <td>
     <img src="images/3_LR_Set5_004.png"/>
    </td>
    <td>
     <img src="images/3_HR_Set5_004.png"/>
    </td>
    <td>
     <img src="images/3_SR_Set5_004.png"/>
    </td>
  </tr>
</table>

- Set14
<table>
  <tr>
    <td>
     <img src="images/3_LR_Set14_001.png"/>
    </td>
    <td>
     <img src="images/3_HR_Set14_001.png"/>
    </td>
    <td>
     <img src="images/3_SR_Set14_001.png"/>
    </td>
  </tr>
</table>

- BSD100
<table>
  <tr>
    <td>
     <img src="images/3_LR_BSD100_063.png"/>
    </td>
    <td>
     <img src="images/3_HR_BSD100_063.png"/>
    </td>
    <td>
     <img src="images/3_SR_BSD100_063.png"/>
    </td>
  </tr>
</table>
