import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class VGG_Loss(nn.Module):
    def __init__(self, loss_type):
        super(VGG_Loss, self).__init__()
        vgg_features = torchvision.models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        conv_index = loss_type[-2:]

        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index == '33':
            self.vgg = nn.Sequential(*modules[:16])
        elif conv_index == '44':
            self.vgg = nn.Sequential(*modules[:26])
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35])
        self.vgg = nn.DataParallel(self.vgg).cuda()

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229, 0.224, 0.225)
        self.sub_mean = MeanShift(vgg_mean, vgg_std)
        self.vgg.requires_grad = False
        self.conv_index = conv_index

    def forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x

        def _forward_all(x):
            feats = []
            x = self.sub_mean(x)
            for module in self.vgg.module:
                x = module(x)
                feats.append(x)
            return feats

        if self.conv_index == 'P':
            vgg_sr_feats = _forward_all(sr)
            with torch.no_grad():
                vgg_hr_feats = _forward_all(hr.detach())
            loss = 0
            for i in range(len(vgg_sr_feats)):
                loss_f = F.mse_loss(vgg_sr_feats[i], vgg_hr_feats[i])
                # print(loss_f)
                loss += loss_f
            # print()
        else:
            vgg_sr = _forward(sr)
            with torch.no_grad():
                vgg_hr = _forward(hr.detach())
            loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss
    """
    def forward(self, output, gt):
        vgg_output = self.vgg16_conv_4_3(output)
        with torch.no_grad():
            vgg_gt = self.vgg16_conv_4_3(gt.detach())

        loss = F.mse_loss(vgg_output, vgg_gt)

        return loss
    """


class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()

        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_name = loss.split('*')
            if loss_name == 'L1':
                loss_function = nn.L1Loss()
            elif loss_name.find('VGG') >= 0:
                loss_function = VGG_Loss(loss_name[3:])
            elif loss_name == 'MSE':
                loss_function = nn.MSELoss()
            self.loss.append({
                'name': loss_name,
                'weight': float(weight),
                'function': loss_function
            })

        if len(self.loss) > 1:
            self.loss.append({'name': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                self.loss_module.append(l['function'])

        device = torch.device('cuda' if args.cuda else 'cpu')
        self.loss_module.to(device)
        if torch.cuda.is_available():  # and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(self.loss_module)

    def forward(self, output, gt):
        loss = 0
        losses = {}
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                _loss = l['function'](output, gt)
                effective_loss = l['weight'] * _loss
                losses[l['name']] = effective_loss
                loss += effective_loss

        return loss, losses