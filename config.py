import datetime
import argparse
import numpy as np


##### import path #####
## Dataset path for UCF101
# datasetPath = 'D:/KIEN/Dataset/UCF101/UCF101_Dataset/'

## Dataset path for Vimeo_90K
datasetPath = 'F:/vimeo_triplet'
# datasetPath = 'D:/KIEN/Dataset/vimeo_septuplet/'

## Dataset path for SNU-FILM
# datasetPath = 'D:/KIEN/Dataset/SNUFILM/'

arg_lists = []
parser = argparse.ArgumentParser(description='My Network')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

### Argument for RSTCANet ###
network_arg = add_argument_group('Network')
network_arg.add_argument('--modelName', type=str, default='RSTSCANet', help='model architecture',
                         choices=['RSTSCANet'])
network_arg.add_argument('--nb_frame', type=int, default=2)
network_arg.add_argument('--joinType', choices=['concat', 'add', 'none'], default='concat')

### DATASET ###
dataset_arg = add_argument_group('Dataset')
dataset_arg.add_argument('--datasetName', type=str, default='Vimeo_90K',
                         choices=['UCF101', 'Vimeo_90K', 'VimeoSepTuplet', 'Snufilm'])
dataset_arg.add_argument('--datasetPath', default=datasetPath, help='the path of selected datasets')


### DATALOADER ###
dataloader_arg = add_argument_group('Dataloader')
dataloader_arg.add_argument('--batch_size', type=int, default=8, help='train batch size')
dataloader_arg.add_argument('--val_batch_size', type=int, default=64, help='val batch size')
dataloader_arg.add_argument('--test_batch_size', type=int, default=64, help='val batch size')
dataloader_arg.add_argument('--num_workers', '-w', type=int, default=4,
                    help='parallel workers for loading training samples')


### TRAIN/TEST PARAMETERS ###
learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--loss', type=str, default='1*L1', help='loss function for optimization')
learn_arg.add_argument('--checkpoint_dir', type=str,
                       default=None,
                       help='path to checkpoint dir')
learn_arg.add_argument('--resume', action='store_true')
learn_arg.add_argument('--start_epoch', type=int, default=0)
learn_arg.add_argument('--max_epoch', '-e', type=int, default=251, help='Number of epochs to train')
learn_arg.add_argument('--best_psnr', type=float, default=0.)


### OPTIMIZER ###
optimizer_arg = add_argument_group('Optimizer')
optimizer_arg.add_argument('--opt_name', type=str, default='ADAM', help='Optimizer used for training')
optimizer_arg.add_argument('--lr', type=float, default=1e-4, help='learning rate')
optimizer_arg.add_argument('--weight_decay', type=float, default=1e-6, help='the weight decay for whole network ')
optimizer_arg.add_argument('--beta1', type=float, default=0.9)
optimizer_arg.add_argument('--beta2', type=float, default=0.99)
optimizer_arg.add_argument('--eps', type=float, default=1e-8)

### IMAGES SIZE ###
img_arg = add_argument_group('Image Inf')
img_arg.add_argument('--height', default=256, type=int, help='image height')
img_arg.add_argument('--width', default=256, type=int, help='image width')
img_arg.add_argument('--channel', default=3, type=int, help='image channel')


### SCHEDULER ###
scheduler_arg = add_argument_group('Scheduler')
scheduler_arg.add_argument('--scheduler', type=str, default='ReduceLROnPlateau', help='the name of scheduler is being used')
scheduler_arg.add_argument('--patience', type=int, default=5, help='the patience of reduce on plateou')
scheduler_arg.add_argument('--factor', type=float, default=0.5, help='the factor of reduce on plateou')
scheduler_arg.add_argument('--min_lr', type=float, default=0, help='the min lr of reduce on plateou')


### MISC ###
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--random_seed', type=int, default=1, help='random seed (default: 1)')
misc_arg.add_argument('--cuda', default=True, type=bool, help='use cuda or not')
misc_arg.add_argument('--use_cudnn', default=1, type=int, help='use cudnn or not')
misc_arg.add_argument('--uid', type=str, default=None, help='unique id for the training')
misc_arg.add_argument('--use_tensorboard', action='store_true')
misc_arg.add_argument('--num_gpu', type=int, default=1)
misc_arg.add_argument('--log_iter', type=int, default=200)


misc_arg.add_argument('--create_reference_images', action='store_true',
                      help='create reference frame at at desired size or not')
misc_arg.add_argument('--reference_folder', type=str, default='Reference_Images',
                      help='the name of folder contains reference images')
misc_arg.add_argument('--ref_counter', type=int, default=0,
                      help='counter reference images')

misc_arg.add_argument('--result_images_folder', type=str, default='Result_Images',
                      help='the name of folder contains result images')
misc_arg.add_argument('--out_counter', type=int, default=0,
                      help='counter result images')
misc_arg.add_argument('--graph_folder', type=str, default='graph',
                      help='the name of folder contains graph results (loss, psnr, ssim,...)')
misc_arg.add_argument('--save_path', type=str, default='', help='the output dir of weights')
misc_arg.add_argument('--timestamp', type=str, default='', help='time when execute the code')
misc_arg.add_argument('--loss_list_data', type=list, default=[], help='list to save loss value in csv file')

def get_args():
    """Parses all of the arguments above
    """
    args, unparsed = parser.parse_known_args()
    if args.num_gpu > 0:
        setattr(args, 'cuda', True)
    else:
        setattr(args, 'cuda', False)
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed



