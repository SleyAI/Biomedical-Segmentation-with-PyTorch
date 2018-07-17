# Biomedical Segmentation with PyTorch

Pixel-wise segmentation for biomedical images using [Pytorch][Pytorch].
This project is part of my Bachelor Thesis and will be extended in the upcoming weeks.


### Dataset

[//]: # (Image References)

[images_and_masks]: etc/Images.png

![alt text][images_and_masks]


The code was tested on a biomedical breast cancer dataset which can't be provided due to data privacy.
A very similar dataset can be found at [bioimage.ucsb.edu](https://bioimage.ucsb.edu/research/bio-segmentation)
Note: The images have a different size than the images I'm using. The network achitecture needs to be tuned a little bit to work with the data.


### Architectures

I will implement the following architectures:

- [x] [U-Net](https://arxiv.org/abs/1505.04597)
- [ ] [SegNet](https://arxiv.org/abs/1511.00561)
- [ ] [PSPNet](https://arxiv.org/abs/1612.01105)


### Required packages

```
torch
torchvision
numpy
PIL
pydensecrf
```

The code is based on Python 3.6


### Folder Structure

	.
    ├── data                    # training dataset
    ├── val                     # validation set
    ├── checkpoints             # checkpoints to store the model

Data and Val folder need to contain a .txt file with filenames of the data. The "create_txt" script can be used to generate these.


### Usage

There are multiple parameters which can be used:

main.py
```
-e	# number of epochs to train
-l	# learning rate
-g	# use this parameter to utilize your GPU
-c	# load a pretrained model
```

predict.py
```
-m	# path to the pretrained model / checkpoint
-i	# input image to predict
-o	# filename of the output image
-c	# GPU support is enabled by default. Use this parameter to predict on CPU
```

There are some more parameters which can be useful.


### Evaluation

Further evaluation needs to be done in the future.
For now the net reached an Accuracy of 87% (Dice Coefficient of 0.87) on a small validation set.


### Some notes about hardware

I'm using a Nvidia GTX 1070 (8GB VRAM) to train the net. The highest memory usage I could observe was about 5,5GB.
The size of the images is 510x512 which will be downscaled to 255x256. Downscaling even further will lower the memory needed.

Training took about 1 1/2 hours.


[Pytorch]: http://pytorch.org