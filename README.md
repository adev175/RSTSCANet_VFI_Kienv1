# RSTSCANet_VFI
This is the repo for my research about Video Frame Interpolation.

## Dependencies
Current version is tested on: 
* python == 3.8
* numpy == 1.19.5
* [pytorch](https://pytorch.org/) == 1.10.1, torchvision == 0.11.2, cuda == 11.3
* cupy == 10.6.0
* tensorboard == 2.11.0
* einops == 0.4.1

## Train
* We use [Vimeo90K Triplet dataset](http://toflow.csail.mit.edu/) for training + testing
* Then train RSTSCANet as below:
'''
python main.py --datasetName Vimeo_90K --datasetPath <dataset_root> --batch_size <batch_size>
'''

## Test
After training, you can evaluate the model with following command:
'''
'''
