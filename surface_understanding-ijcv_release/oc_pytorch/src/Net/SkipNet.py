# Pix2Pix : https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

import sys
sys.path.append('/scratch/local/ssd/ow/code/oc_pytorch/src/')
import utils
print(dir(utils))
from utils.Globals import *

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
###############################################################################
# Functions
###############################################################################



def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[], modify_unet=False, unit=-1, increase=0):
    print("USING Pix2Pix")
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGeneratorBetterUpsampler(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda()
    init_weights(netG, init_type=init_type)
    return netG



def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

# Defines the SkipNetWork
class Pix2PixModel(nn.Module):
    def __init__(self, loss_type=None, num_variables=None, modify_unet=False, unit=-1, increase=0):
        super(Pix2PixModel, self).__init__()

        self.netG = define_G(3, 1, 64, 'unet_256', 'batch', True, 'xavier', [0], modify_unet=modify_unet, unit=unit, increase=increase)

    def forward(self, *cycles):
        # First one
        xc = self.netG(cycles[0], *cycles[1:])
        return xc



# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], modify_unet=False, unit=-1, increase=0):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlockOutput(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, modify_unet=modify_unet, unit=unit, increase=increase)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlockOutput(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlockOutput(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockOutput(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockOutput(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockOutput(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, x, *views):
        # if self.gpu_ids and isinstance(x.data, torch.cuda.FloatTensor):
        #     output, output_orig = nn.parallel.data_parallel(self.model, (x, views), self.gpu_ids)
        #     return output, output_orig
        # else:
        return self.model(x, *views)

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck

class UnetGeneratorBetterUpsampler(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGeneratorBetterUpsampler, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlockBetterUpsampler(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        pow_2 = 128
        for i in range(num_downs - 5):
            pow_2 = pow_2 / 2
            if INPUT_ANGLES:
                print("INPUT ANGLES")
                if i == 1:
                    has_angle = True
                else:
                    has_angle = False
            else:
                print("NO INPUT ANGLES")
                has_angle = False

            unet_block = UnetSkipConnectionBlockBetterUpsampler(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, has_angle=has_angle, size_down=256/pow_2)

        unet_block = UnetSkipConnectionBlockBetterUpsampler(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockBetterUpsampler(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockBetterUpsampler(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockBetterUpsampler(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, x, *views):
        # if self.gpu_ids and isinstance(x.data, torch.cuda.FloatTensor):
        #     output, output_orig = nn.parallel.data_parallel(self.model, (x, views), self.gpu_ids)
        #     return output, output_orig
        # else:
        t = self.model(x, *views)
        return t

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlockOutput(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 modify_unet=False, unit=-1,increase=0):
        super(UnetSkipConnectionBlockOutput, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.use_dropout = use_dropout

        print("USING DROPOUT: %s" + str(self.use_dropout))

        self.modify_unet = False
        self.unit = unit
        self.increase = increase

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            self.up = nn.Sequential(*up)
            self.down = nn.Sequential(*down)
            self.submodule = submodule
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            self.up = nn.Sequential(*up)
            self.down = nn.Sequential(*down)
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            
            self.upnorm = upnorm
            self.upconv = upconv
            self.uprelu = uprelu
            self.submodule = submodule
            self.down = nn.Sequential(*down)

            self.dropout = nn.Dropout(0.5)

        #self.conv_models = nn.Conv2d(2 * 482, 482, kernel_size=1)


    def forward(self, x_orig, *other_inputs):
        # Assuming that the first set of units are viewpoint; the rest are 3D
        # Then we can concat (max / sum / whatever these parts)
        # And the rest is only the viewpoint
        x_fv = self.down(x_orig)

        others_fv = []
        for input_vec in other_inputs:
            others_fv += [self.down(input_vec)]

        if self.innermost:
            vp = x_fv[:,0:30,:,:]
            model_3d = x_fv[:,30:,:,:]

            for input_fv in others_fv:
                t_model_3d = input_fv[:,30:,:,:]
                model_3d = torch.cat([model_3d, t_model_3d], 2)

            model_3d = model_3d.mean(dim=2, keepdim=True)
            #model_3d = self.conv_models(model_3d)
            x_fv = torch.cat([vp, model_3d], 1)

            if self.modify_unet:
                x_fv[self.unit] = x_fv[self.unit] + self.increase
            
            x = self.up(x_fv)
            return torch.cat([x, x_orig], 1), x_fv
        if self.outermost:
            x, x_fv = self.submodule(x_fv, *others_fv)
            x = self.up(x)
            return x, x_fv
        else:
            x, x_fv = self.submodule(x_fv, *others_fv)
            if self.use_dropout:
                x = self.dropout(self.upnorm(self.upconv(self.uprelu(x))))
            else:
                x = self.upnorm(self.upconv(self.uprelu(x)))
            return torch.cat([x,  x_orig], 1), x_fv

from utils.Globals import *

class ConditionedFilter(nn.Module):
    def __init__(self, outer_nc, inner_nc):
        super(ConditionedFilter, self).__init__()
        down = [nn.Conv2d(2, 32, kernel_size=1), nn.ReLU()]
        self.theta_conv1 = nn.Sequential(*down)
        down = [nn.Conv2d( 32, (outer_nc+1) * inner_nc, kernel_size=1),  nn.ReLU()]
        self.theta_conv2 = nn.Sequential(*down)

        self.outer_nc = outer_nc
        self.inner_nc = inner_nc

    def forward(self, input_tensor, angle):
        angle = self.theta_conv1(angle)
        angle = self.theta_conv2(angle)

        w = input_tensor.size(2)

        angle = angle.view((angle.size(0), self.outer_nc+1, self.inner_nc)).contiguous()
        input_tensor = input_tensor.view((input_tensor.size(0), input_tensor.size(1), input_tensor.size(2) * input_tensor.size(3)))
        
        input_tensor = torch.baddbmm(angle[:,-1,:].unsqueeze(2), angle[:,:-1,:], input_tensor).view(input_tensor.size(0), input_tensor.size(1), w, w).contiguous()
        return input_tensor

class MultiplicativeFilter(nn.Module):
    def __init__(self, num_channels, w, h):
        super(MultiplicativeFilter, self).__init__()
        down = [nn.Conv2d(2, 64, kernel_size=1), nn.ReLU()]
        self.theta_conv1 = nn.Sequential(*down)
        down = [nn.Conv2d( 64, (num_channels+1) * w * h, kernel_size=1),  nn.ReLU()]
        self.theta_conv2 = nn.Sequential(*down)
        self.num_channels = num_channels
        self.w = w
        self.h = h

    def forward(self, input_tensor, angle):
        angle = self.theta_conv1(angle)
        angle = self.theta_conv2(angle)
        angle = angle.view((angle.size(0), input_tensor.size(1)+1, input_tensor.size(2), input_tensor.size(3))).contiguous()
        input_tensor = angle[:,:-1,:,:] * input_tensor + angle[:,-2:-1,:,:]
        
        return input_tensor


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlockBetterUpsampler(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, has_angle=False, size_down=None):
        super(UnetSkipConnectionBlockBetterUpsampler, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.has_angle = has_angle
        self.use_dropout = use_dropout

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc


        if has_angle:
            fv_1 = nn.Conv2d(2,32,1)
            fv_2 = nn.Conv2d(32,32,1)

            angle = [fv_1, nn.ReLU(), fv_2, nn.ReLU()]
            self.angle_fv = nn.Sequential(*angle)
            input_nc = input_nc + 32
            print('angle', innermost, outermost, inner_nc, outer_nc)

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)


        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=1,
                                        padding=1, bias=use_bias)
            down = [downconv]
            up = [uprelu, upsample, upconv, nn.Tanh()]
            self.up = nn.Sequential(*up)
            self.down = nn.Sequential(*down)
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')

            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(inner_nc, outer_nc,
                                        kernel_size=3, stride=1,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv, upnorm]
            self.up = nn.Sequential(*up)
            self.down = nn.Sequential(*down)
        else:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=1,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]

            self.up = nn.Sequential(*up)
            self.down = nn.Sequential(*down)
            self.dropout = nn.Dropout(0.5)
        self.submodule =submodule


    def forward(self, *inputs_orig):
        num_views = len(inputs_orig) / 3
        inputs = [0] * len(inputs_orig)
        for i in range(0, num_views):
            if self.has_angle:
                inputs[3*i+1] = self.angle_fv(inputs_orig[3*i+1]).repeat(1,1,inputs_orig[3*i].size(2),inputs_orig[3*i].size(3))
                inputs[3*i] = torch.cat((inputs_orig[3*i], inputs[3*i+1]), 1)
                inputs[3*i] = self.down(inputs[3*i])
            else:
                inputs[3*i] = self.down(inputs_orig[3*i])
                inputs[3*i+1] = inputs_orig[3*i+1]
            inputs[3*i+2] = inputs_orig[3*i+2]

        if self.innermost:
            model_3d = inputs[0]
            for i in range(1, num_views):
                t_model_3d = inputs[3*i]
                model_3d = torch.cat([model_3d, t_model_3d], 2)
            
            if USE_AVERAGE:
                central_fv = model_3d.mean(dim=2, keepdim=True)
            else:
                central_fv = model_3d.max(dim=2, keepdim=True)[0]

            central_fv_up = self.up(central_fv)
            results = [0] * num_views
            for i in range(0, num_views):
                results[i] = torch.cat([inputs_orig[3*i], central_fv_up], 1)

            return tuple(results) + tuple([central_fv])
        if self.outermost:
            results_t = list(self.submodule(*inputs))
            results = [0] * len(results_t)
            for i in range(0, len(results)-1):
                results[i] = self.up(results_t[i]) * 5
            return tuple(results[:-1]) + tuple([results_t[-1]])
        else:
            results_t = list(self.submodule(*inputs))
            results = [0] * len(results_t)

            for i in range(0, len(results)-1):
                if self.use_dropout:
                    t = self.dropout(self.up(results_t[i]))
                else:
                    t = self.up(results_t[i])
                results[i] = torch.cat([inputs_orig[3*i], t], 1)
            return tuple(results[:-1]) + tuple([results_t[-1]])



# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.innermost:
            xc_orig = self.model[2](self.model[1](self.model[0](x)))
            x_new = self.model[3](xc_orig)
            x_new = self.model[4](x_new)
            self.model.fc = xc_orig
            return torch.cat([x, x_new], 1)
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x,  self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
