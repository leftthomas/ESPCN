# Super Resolution using an efficient sub-pixel convolutional neural network
A PyTorch implementation of ESPCN based on CVPR2016 paper [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158)

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
Download the datasets from [here](), and then extract it into `data` directory. Finally run
```
python data_utils.py
```
to generate train and val datasets from VOC2012 with different upscale factors(2、3、4、8).

### Test Dataset
The test dataset are sampled from 

| **Set 5** |  [Bevilacqua et al. BMVC 2012](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)  | [16.1 MB](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip)|

| **Set 14** |  [Zeyde et al. LNCS 2010](https://sites.google.com/site/romanzeyde/research-interests)  | [86.0 MB](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip)|

| **BSD 100** | [Martin et al. ICCV 2001](https://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) | [568 MB](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip)|

| **Sun-Hays 80** | [Sun and Hays ICCP 2012](http://cs.brown.edu/~lbsun/SRproj2012/SR_iccp2012.html) | [311 MB](https://uofi.box.com/shared/static/rirohj4773jl7ef752r330rtqw23djt8.zip)|

| **Urban 100** | [Huang et al. CVPR 2015](https://sites.google.com/site/jbhuang0604/publications/struct_sr)  | [ 1.14 GB](https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip)|

Download the dataset from [here](), and then extract it into `data` directory.

## Usage

### Train

```
python -m visdom.server & python train.py
```
Visdom now can be accessed by going to `127.0.0.1:8097` in your browser, or your own host address if specified.

### Test
```
python test.py
```
The output high resolution images are on `results` directory.

## Benchmarks
Highest accuracy was 99.57% after 30 epochs. The model may achieve a higher accuracy as shown by the trend of the loss/accuracy graphs below.
<table>
  <tr>
    <td>
     <img src="results/train_loss.png"/>
    </td>
    <td>
     <img src="results/test_loss.png"/>
    </td>
  </tr>
</table>
<table>
  <tr>
    <td>
     <img src="results/train_acc.png"/>
    </td>
    <td>
     <img src="results/test_acc.png"/>
    </td>
  </tr>
</table>

The confusion matrix of the digit numbers are showed below.
<img src="results/confusion_matrix.png"/>

The reconstructions of the digit numbers are showed at right and the ground truth at left.
<table>
  <tr>
    <td>
     <img src="results/ground_truth.jpg"/>
    </td>
    <td>
     <img src="results/reconstruction.jpg"/>
    </td>
  </tr>
</table>

Default PyTorch Adam optimizer hyperparameters were used with no learning rate scheduling. Epochs with batch size of 100 takes ~2 minutes on a NVIDIA GTX 1070 GPU. 

