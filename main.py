import os
import numpy as np
import time
import datetime
from tqdm import tqdm

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import utils
import config
from loss_function import Loss

##### Parse CmdLine Arguments #####
args, unparsed = config.get_args()
cwd = os.getcwd()

# Device configuration
device = torch.device('cuda' if args.cuda else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True       # True when input is constant -> cudnn will look for the optimal set of algorithms
                                                # for that particular configuration (which takes some time) -> faster runtime.

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

##### Load Dataset #####
train_set, val_set, train_loader, val_loader = utils.load_dataset(
    args.datasetName, args.datasetPath, args.batch_size, args.val_batch_size, args.num_workers
)

##### Build Model #####
from rstsca_model.RSTSCANet import RSTSCANet
model = RSTSCANet(n_inputs=args.nb_frame, joinType=args.joinType)
model = model.to(device)
#print(model)

##### Define Loss, Optimizer and Scheduler #####
criterion = Loss(args)

if args.opt_name == 'ADAM':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)
elif args.opt_name == 'RADAM':
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr,
                                  betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)
elif args.opt_name == 'ADAMAX':
    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
else:
    raise NotImplementedError('Optimizer is not implemented')

if args.scheduler == 'ReduceLROnPlateau':
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=args.factor,
                                           patience=args.patience, min_lr=0, verbose=True)
elif args.scheduler == 'MultiStepLR':
    scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30, 80, 120], gamma=0.1)
else:
    raise NotImplementedError('Scheduler is not implemented')

LOSS_0 = 0

def train(args, epoch, writer):
    global LOSS_0
    torch.cuda.empty_cache()
    losses, psnrs, ssims = utils.init_meters(args.loss, reset_loss=True)
    model.train()
    criterion.train()

    start = time.time()
    for i, (images, gt_image, _) in enumerate(train_loader):
        # Build input batch
        images = [img_.to(device) for img_ in images]
        gt = gt_image.to(device)

        # Forward
        optimizer.zero_grad()
        out = model(images)
        loss, loss_specific = criterion(out, gt)

        # Save loss values
        for k, v in losses.items():
            if k != 'total':
                v.update(loss_specific[k].item())
        losses['total'].update(loss.item())
        if LOSS_0 == 0:
            LOSS_0 = loss.data.item()

        # Backward (+ grad clip) - if loss explodes, skip current iteration
        loss.backward()
        if loss.data.item() > 10.0 * LOSS_0:
            print(max(p.grad.data.abs().max() for p in model.parameters()))
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        # Calc metrics
        if i % args.log_iter == 0:
            utils.eval_metrics(out, gt, psnrs, ssims)

            print('Train Epoch {}: [{}/{}]\tLoss: {:.6f}\tPSNR: {:.4f}\t SSIM: {:.3f} \tTime({:.2f})'.format(
                epoch, i, len(train_loader), losses['total'].avg, psnrs.avg, ssims.avg.item(), time.time() - start))

            # Tensorboard
            utils.log_tensorboard(writer, losses, psnrs.avg, ssims.avg,
                                  optimizer.param_groups[0]['lr'], epoch * len(train_loader) + i, 'train')

            # Reset metrics
            losses, psnrs, ssims= utils.init_meters(losses, reset_loss=False)

    # Calculatet execution train time
    train_time_elapsed = time.time() - start

    # save train metrics to csv
    if epoch % 2 == 0:
        utils.save_metrics(args, os.path.join(args.save_path, args.graph_folder), epoch,
                           losses['total'].avg, None, None,
                           optimizer.param_groups[0]['lr'], train_time_elapsed, 'train')


def validation(args, epoch, writer):
    torch.cuda.empty_cache()
    losses, psnrs, ssims = utils.init_meters(args.loss, reset_loss=True)
    model.eval()
    criterion.eval()

    args.out_counter = 0  # Reset the output images index
    start = time.time()
    with torch.no_grad():
        loop = tqdm(enumerate(val_loader), total=len(val_loader))
        for i, (images, gt_image, _) in loop:
            # Build input batch
            images = [img_.to(device) for img_ in images]
            gt = gt_image.to(device)

            # Forward
            out = model(images)
            loss, loss_specific = criterion(out, gt)

            # Save loss values
            for k, v in losses.items():
                if k != 'total':
                    v.update(loss_specific[k].item())
            losses['total'].update(loss.item())

            # Calc metrics
            utils.eval_metrics(out, gt, psnrs, ssims)

            # Tensorboard
            if i % args.log_iter == 0:
                utils.log_tensorboard(writer, losses, psnrs.avg, ssims.avg.item(),
                                      optimizer.param_groups[0]['lr'], epoch * len(train_loader) + i, 'val')

            # Save result images
            # if epoch % 15 == 0:
            #     epoch_path = os.path.join(args.save_path, args.result_images_folder, 'Epoch_' + str(epoch))
            #     utils.save_batch_images(args, out, gt, epoch_path)

            # update progress bar
            loop.set_description("(Val)")
            loop.set_postfix(loss=loss.item())

        # Calculatet execution validation time
        val_time_elapsed = time.time() - start

        # save val metrics to csv
        if epoch % 2 == 0:
            utils.save_metrics(args, os.path.join(args.save_path, args.graph_folder), epoch,
                               losses['total'].avg, psnrs.avg, ssims.avg.item(),
                               optimizer.param_groups[0]['lr'], val_time_elapsed, 'val')

        print('Validating Results: \tValidation Loss: {:.6f}\tVal Time: {:.2f}'
              '\tPSNR: {:.4f}\tSSIM: {:.3f}'.format(losses['total'].avg, val_time_elapsed,
                                                    psnrs.avg, ssims.avg.item()))

        return losses['total'].avg, psnrs.avg

""" Entry Point """
def main(args):
    # If train model from beginning, create save_path and save in args
    if not args.resume:
        print('Start training from the beginning')
        unique_id = str(np.random.randint(0, 100000))
        args.uid = unique_id
        args.timestamp = datetime.datetime.today().strftime("%Y%m%d_%H%M")
        args.save_path = './Results/' + args.timestamp + '_' + args.uid
    else:   # Use pretrained model
        # args.checkpoint_dir = ['D:', 'KIEN', 'My Model', 'Results', '20220512_2001_37155', 'My_Network_NestedUNet_epoch65.pth']
        # args.uid = args.checkpoint_dir.split('\\')[4].split('_')[2]
        # args.timestamp = '_'.join(args.checkpoint_dir.split('\\')[4].split('_')[0: 2])
        args.save_path = './Results/'

        utils.load_checkpoint(args, args.checkpoint_dir, model, optimizer, scheduler)

    # Create some folders
    utils.makedirs(args.save_path)      # save_path folder
    utils.makedirs(os.path.join(args.save_path, args.graph_folder))     #csv folder

    # Create Summary Writer
    writer = SummaryWriter(f"runs2/{args.save_path}")

    # Write information into args.txt
    with open(os.path.join(args.save_path, 'arg.txt'), 'w') as f:
        print(args, file=f)
        f.close()

    # Write essisials information into readme.txt
    with open(os.path.join(args.save_path, 'readme.txt'), 'w', newline='') as f:
        f.write(f"""model Name: {args.modelName}
batch_size: {args.batch_size}
val_batch_size: {args.val_batch_size}
dataset: {args.datasetName}
img_h: {args.height}
img_w: {args.width}
train_num: {len(train_set)}
val_num: {len(val_set)}
number_workers: {args.num_workers}
epochs: {args.max_epoch}
loss: {args.loss}
lr: {args.lr}
optimizer: {args.opt_name}
weight_decay: {args.weight_decay}
scheduler: {args.scheduler}
factor: {args.factor}
patience: {args.patience}
min_lr: {args.min_lr}
model parameters: {str(utils.count_network_parameters(model))}""")
        f.close()

    print("EPOCH is: " + str(int(len(train_set) / args.batch_size)))
    print("Num of EPOCH is: " + str(args.max_epoch))
    print("The id of this interp network is " + str(args.uid))
    print("Num. of flow model parameters is " + str(utils.count_network_parameters(model)))
    print('{} samples found, {} train samples and {} test samples '.format(len(val_set) + len(train_set),
                                                                           len(train_set), len(val_set)))
    print("----------------------------------------------------------------")

    ##########################
    # TRAINING START FROM HERE
    ##########################

    for epoch in range(args.start_epoch, args.max_epoch):
        print(f"Epoch [{epoch}/{args.max_epoch}]")
        print("Learning rate: " + str(optimizer.param_groups[0]['lr']))

        # Run training
        train(args, epoch, writer)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'best_psnr': args.best_psnr,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': optimizer.param_groups[0]['lr'],
            'scheduler': scheduler.state_dict(),
        }
        utils.save_checkpoint(args, checkpoint, is_best=False, epoch=epoch)

        # Run validating
        val_loss, val_psnr = validation(args, epoch, writer)

        # Save best model
        is_best = val_psnr > args.best_psnr
        if is_best:
            args.best_psnr = max(val_psnr, args.best_psnr)
            utils.save_checkpoint(args, checkpoint, is_best=is_best, epoch=epoch)
            print("Best Weights updated for decreased psnr\n")
        else:
            print("Weights Not updated for undecreased psnr\n")

        # schedule the learning rate
        scheduler.step(val_loss)

        print("*********Finish Training********")
        print("----------------------------------------------------------------")

if __name__ == "__main__":
    # print(torch.__version__)
    main(args)









