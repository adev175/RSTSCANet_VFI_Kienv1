import os
import shutil
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import transforms
from torchvision.utils import save_image

from pytorch_msssim import ssim as evaluator_SSIM
from pytorch_msssim import ssim_matlab as calc_ssim

##########################
# Training Helper Functions for making main.py clean
##########################

def save_checkpoint(args, state, is_best, epoch):
    """
        state: checkpoint we want to save
        is_best: is this the best checkpoint; min validation loss
    """

    # Create args.save_path if not exist
    makedirs(args.save_path)

    filename = 'epoch' + str(epoch) + '.pth'
    previousfilename = 'epoch' + str(epoch-1) + '.pth'

    # Delete previous checkpoint
    remove_file(os.path.join(args.save_path, '_'.join((args.modelName, previousfilename))))

    # save checkpoint data to the path given, filename_path
    filename_path = os.path.join(args.save_path, '_'.join((args.modelName, filename)))
    torch.save(state, filename_path)
    # if it is a best model, min validation loss
    if is_best:
        best_path = os.path.join(args.save_path, '_'.join(
            (args.modelName, 'best.pth')))
        # copy that checkpoint file to best path given, best_path
        shutil.copyfile(filename_path, best_path)

def load_checkpoint(args, checpoint_path, model, optimizer, scheduler):
    """
        checkpoint_path: path to save checkpoint
        model: model that we want to load checkpoint parameters into
        optimizer: optimizer we defined in previous training
        scheduler: scheduler we defined in previous training
    """
    print('Loading checkpoint from %s' %checpoint_path)
    # Load check point
    checkpoint = torch.load(checpoint_path)

    args.start_epoch = checkpoint['epoch']
    args.best_psnr = checkpoint['best_psnr']
    args.lr = checkpoint['lr']

    model.load_state_dict(checkpoint['state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer, scheduler

def load_dataset(datasetName, datasetPath, batch_size, val_batch_size, num_workers):
    if datasetName == 'UCF101':
        from datasets.ucf101 import ucf101

        train_set = ucf101.UCF101(root=datasetPath + 'train1', is_training=True)
        val_set = ucf101.UCF101(root=datasetPath + 'test1', is_training=False)
    elif datasetName == 'Vimeo_90K':
        from datasets.vimeo_90K.vimeo_90K import Vimeo_90K

        train_set = Vimeo_90K(root=datasetPath, is_training=True)
        val_set = Vimeo_90K(root=datasetPath, is_training=False)
    elif datasetName == 'VimeoSepTuplet':
        from datasets.vimeo_90K.vimeo_90K import VimeoSepTuplet

        train_set = VimeoSepTuplet(root=datasetPath, is_training=True, mode='full')
        val_set = VimeoSepTuplet(root=datasetPath, is_training=False, mode='full')
    else:
        raise NotImplementedError('Training / Testing for this dataset is not implemented')

    train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, pin_memory=True,
                                             batch_size=val_batch_size,
                                             num_workers=num_workers,
                                             shuffle=False, drop_last=True)


    return train_set, val_set, train_loader, val_loader

##########################
# PSNR and SSIM Calculation
##########################
class EvalPSNR(object):
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2) + 1e-8
        return 20 * torch.log10(255.0 / torch.sqrt(mse))

def calc_psnr(pred, gt):
    diff = (pred - gt).pow(2).mean() + 1e-8
    return -10 * math.log10(diff)

##########################
# Evaluations
##########################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def init_losses(loss_str):
    loss_specifics = {}
    loss_list = loss_str.split('+')
    for l in loss_list:
        _, loss_name = l.split('*')
        loss_specifics[loss_name] = AverageMeter()
    loss_specifics['total'] = AverageMeter()
    return loss_specifics

def init_meters(loss_str, reset_loss=True):
    if reset_loss:
        losses = init_losses(loss_str)
    else:
        losses = loss_str
    psnrs = AverageMeter()
    ssims = AverageMeter()
    return losses, psnrs, ssims

def quantize(img, rgb_range=255.):
    return img.mul(255. / rgb_range).round()

def eval_metrics(im_pred, im_gt, psnrs, ssims):
    # PSNR should be calculated for each image, since sum(log) =/= log(sum).
    for i in range(im_gt.size()[0]):
        psnr = calc_psnr(im_pred[i], im_gt[i])
        psnrs.update(psnr)

        ssim = calc_ssim(im_pred[i].unsqueeze(0).clamp(0, 1), im_gt[i].unsqueeze(0).clamp(0, 1),
                         val_range=1.)
        ssims.update(ssim)

def save_metrics(args, save_path, epoch, loss, psnr, ssim, lr, time, mode='train'):
    # Create list for train and val loss,
    # cannot write train and val loss at a same row
    if mode == 'train':
        args.loss_list_data.append(lr)
        args.loss_list_data.append(epoch)
    args.loss_list_data.append(loss)
    args.loss_list_data.append(time)

    if mode == 'val':
        with open(os.path.join(save_path, 'val_PSNR_SSIM.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            listdata = []
            listdata.append(epoch)
            listdata.append(round(psnr, 5))
            listdata.append(round(ssim, 5))
            writer.writerow(listdata)

        with open(os.path.join(save_path, 'loss.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(args.loss_list_data)
            args.loss_list_data = []    # reset list

##########################
# ETC
##########################

def makedirs(path):
    if not os.path.exists(path):
        # print("[*] Make directories: {}".format(path))
        os.makedirs(path)   # os.makedirs: creates all the intermediate directories if they don't exist

def remove_file(path):
    if os.path.exists(path):
        # print("[*] Removed: {}".format(path))
        os.remove(path)

def count_network_parameters(model):
    # Calculate model parameters
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    N = sum([np.prod(p.size()) for p in parameters])

    return N

# Tensorboard
def log_tensorboard(writer, losses, psnr, ssim, lr, epoch, mode='train'):
    for k, v in losses.items():
        writer.add_scalar('Loss/%s/%s' % (k, mode), v.avg, epoch)
    writer.add_scalar('PSNR/%s' % mode, psnr, epoch)
    writer.add_scalar('SSIM/%s' % mode, ssim, epoch)
    if mode == 'train':
        writer.add_scalar('lr', lr, epoch)

###########################
###### VISUALIZATIONS
###########################

def distribution_pixels(args, im_pred, im_gt, mode="train", title_name="Distribution of normalized pixels"):
    # Convert tensor to numpy array
    pred_np = im_pred.cpu().detach().numpy()
    gt_np = im_gt.cpu().detach().numpy()

    # Plot prediction image
    plt.subplot(1, 2, 1)
    plt.hist(pred_np.ravel(), bins=50, density=True)
    plt.xlabel("pixel values")
    plt.ylabel("relative frequency")
    plt.title("prediction image")

    # Plot ground trouth image
    plt.subplot(1, 2, 2)
    plt.hist(gt_np.ravel(), bins=50, density=True)
    plt.xlabel("pixel values")
    plt.title("ground truth image")

    # Save plot
    plt.savefig(args.save_path + "/" + mode + ".png")

    plt.suptitle(title_name)
    plt.show()


def save_image(img, path):
    # img : torch Tensor of size (C, H, W)
    q_im = quantize(img.data.mul(255))
    if len(img.size()) == 2:    # grayscale image
        im = Image.fromarray(q_im.cpu().numpy().astype(np.uint8), 'L')
    elif len(img.size()) == 3:
        im = Image.fromarray(q_im.permute(1, 2, 0).cpu().numpy().astype(np.uint8), 'RGB')
    else:
        pass
    im.save(path)

def save_batch_images(args, ims_pred, ims_gt, epoch_path):
    # Check if epoch_path exists
    makedirs(epoch_path)    # ['./Results/20220425_1052_8827', 'Result_Images', 'Epoch_0']

    # Save every image in batch to indicated location
    for j in range(ims_pred.size(0)):
        pred_name = str(args.out_counter) + '_out.png'
        gt_name = str(args.out_counter) + '_gt.png'

        save_image(ims_pred[j, :, :, :], os.path.join(epoch_path, pred_name))
        save_image(ims_gt[j, :, :, :], os.path.join(epoch_path, gt_name))


        # Save reference images at desired size if args.create_reference_images is True and epoch == 0
        """
        if not args.create_reference_images and epoch_path.split('\\')[2].split('_')[1] == 0:
            reference_path = os.path.join(args.save_path, args.reference_folder)
            makedirs(reference_path)

            input1_name = str(args.out_counter) + '_1.png',
            input2_name = str(args.out_counter) + '_2.png'

            save_image(reverse_normalize(input1[j, :, :, :]), os.path.join(reference_path, input1_name))
            save_image(reverse_normalize(input2[j, :, :, :]), os.path.join(reference_path, input2_name))
        """
        args.out_counter += 1



