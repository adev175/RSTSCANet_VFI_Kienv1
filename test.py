import os
import time

import torch
from tqdm import tqdm

from torchvision.utils import save_image
import config
import utils
import math
import torch.nn.functional as F

from torch.utils.data import DataLoader

from check_psnr_ssim import check_psnr_ssim_overall

##### Parse CmdLine Arguments #####
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
args, unparsed = config.get_args()
cwd = os.getcwd()

device = torch.device('cuda' if args.cuda else 'cpu')

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

#### Dataset loading ####
if args.datasetName == 'UCF101':
    from datasets.ucf101 import ucf101
    test_set = ucf101.UCF101(root=args.datasetPath + 'ucf101_interp_ours', is_training=False)
elif args.datasetName == 'Vimeo_90K':
    from datasets.vimeo_90K.vimeo_90K import Vimeo_90K
    test_set = Vimeo_90K(root=args.datasetPath, is_training=False)
elif args.datasetName == 'VimeoSepTuplet':
    from datasets.vimeo_90K.vimeo_90K import VimeoSepTuplet
    test_set = VimeoSepTuplet(root=args.datasetPath, is_training=False, mode='full')
elif args.datasetName == 'Snufilm':
    from datasets.snu_film.snufilm import SNUFILM
    test_set = SNUFILM(data_root=args.datasetPath, mode='hard')

test_loader = torch.utils.data.DataLoader(test_set, pin_memory=True if args.cuda else False,
                                             batch_size=args.test_batch_size,
                                             num_workers=args.num_workers,
                                             shuffle=False, drop_last=True)

print("Building model: %s"%args.modelName)
if args.modelName == 'VFIT_B':
    from my_models.Sep_STS.VFIT_B import UNet_3D_3D
    model = UNet_3D_3D(n_inputs=args.nb_frame, joinType=args.joinType)
elif args.modelName == 'VFIT_S':
    from my_models.Sep_STS.VFIT_S import UNet_3D_3D
    model = UNet_3D_3D(n_inputs=args.nb_frame, joinType=args.joinType)
elif args.modelName == 'RSTCANet':
    from my_models.RSTCANet.rstca import RSTCANet
    model = RSTCANet(args)
elif args.modelName == 'CAIN':
    from my_models.CAIN.cain import CAIN
    model = CAIN()

#model = torch.nn.DataParallel(model).to(device)
model = model.to(device)
print("#params", sum([p.numel() for p in model.parameters()]))
"""
def save_image(recovery, image_name):
    recovery_image = torch.split(recovery, 1, dim=0)
    batch_num = len(recovery_image)

    if not os.path.exists('./results'):
        os.makedirs('./results')

    for ind in range(batch_num):
        utils.save_image(recovery_image[ind], './results/{}.png'.format(image_name[ind]))
"""
def save_batch_images(ims_pred, ims_gt):
    if not os.path.exists('./test_results'):
        os.makedirs('./test_results')
    # Save every image in batch to indicated location
    for j in range(ims_pred.size(0)):
        pred_name = str(args.out_counter) + '_out.png'
        gt_name = str(args.out_counter) + '_gt.png'

        save_image(ims_pred[j, :, :, :], os.path.join('./test_results', pred_name))
        save_image(ims_gt[j, :, :, :], os.path.join('./test_results', gt_name))

        args.out_counter += 1

def to_psnr(rect, gt):
    mse = F.mse_loss(rect, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    psnr_list = [-10.0 * math.log10(mse) for mse in mse_list]
    return psnr_list

def test(args):
    time_taken = []
    losses, psnrs, ssims = utils.init_meters(args.loss, reset_loss=True)
    model.eval()
    args.out_counter = 0

    start = time.time()
    with torch.no_grad():
        for i, (images, gt_image) in enumerate(tqdm(test_loader)):

            images = [img_.cuda() for img_ in images]
            gt = gt_image.cuda()

            torch.cuda.synchronize()

            out = model(images)

            torch.cuda.synchronize()
            time_taken.append(time.time() - start)

            utils.eval_metrics(out, gt, psnrs, ssims)

            save_batch_images(out, gt)

    print("PSNR: %f, SSIM: %fn" %(psnrs.avg, ssims.avg.item()))
    print("Time , ", sum(time_taken)/len(time_taken))

    return psnrs.avg


""" Entry Point """
def main(args):
    
    assert args.checkpoint_dir is not None
    checkpoint = torch.load(args.checkpoint_dir)

    model_dict = model.state_dict()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    test(args)


if __name__ == "__main__":
    main(args)
    check_psnr_ssim_overall(data_path='./test_results/')
