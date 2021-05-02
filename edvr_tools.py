import time
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.utils as utils
import numpy as np
from scipy.signal import convolve2d
from skimage.draw import circle
from math import log10
from skimage import measure
from PIL import Image
# from config import scale, color


def linear2raw(linear):
    h, w, _ = linear.shape
    raw = np.zeros((h, w))

    # red
    raw[::2, ::2] = linear[::2, ::2, 0]
    # green 1
    raw[::2, 1::2] = linear[::2, 1::2, 1]
    # green 2
    raw[1::2, ::2] = linear[1::2, ::2, 1]
    # blue
    raw[1::2, 1::2] = linear[1::2, 1::2, 2]

    return raw


def gamma(image):
    image = image.astype(np.float32) / 255.
    y = (1 + 0.055) * np.power(image, 1 / 2.4) - 0.055
    y[image < 0.0031308] = 12.92 * image[image < 0.0031308]
    return y


# defocus blur
defocusKernelDims = [3, 5, 7, 9, 11]


def defocus_blur_random(img):
    kernelidx = np.random.randint(0, len(defocusKernelDims))
    kerneldim = defocusKernelDims[kernelidx]
    return defocus_blur(img, kerneldim)


def defocus_blur(img, dim):
    imgarray = np.array(img, dtype=np.float64)
    kernel = disk_kernel(dim)

    convolved = np.stack([convolve2d(imgarray[..., channel_id],
                                     kernel, mode='same', boundary='symm')
                          for channel_id in range(3)], axis=2)

    return convolved


def disk_kernel(dim):
    kernelwidth = dim
    kernel = np.zeros((kernelwidth, kernelwidth), dtype=np.float64)
    circleCenterCoord = dim // 2
    circleRadius = circleCenterCoord + 1

    rr, cc = circle(circleCenterCoord, circleCenterCoord, circleRadius)
    kernel[rr, cc] = 1

    if (dim == 3 or dim == 5):
        kernel = adjust(kernel, dim)

    normalizationFactor = np.count_nonzero(kernel)
    kernel = kernel / normalizationFactor
    return kernel


def adjust(kernel, kernelwidth):
    kernel[0, 0] = 0
    kernel[0, kernelwidth - 1] = 0
    kernel[kernelwidth - 1, 0] = 0
    kernel[kernelwidth - 1, kernelwidth - 1] = 0
    return kernel


def to_psnr(derain, gt):
    mse = F.mse_loss(derain, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    # ToTensor scales input images to [0.0, 1.0]
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(derain, gt):
    dehaze_list = torch.split(derain, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in
                      range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    # ssim_list = [measure.compare_ssim(dehaze_list_np[ind], gt_list_np[ind], data_range=1, multichannel=True,
    #                                   gaussian_weights=True) for ind in range(len(dehaze_list))]
    ssim_list = [measure.compare_ssim(dehaze_list_np[ind], gt_list_np[ind], data_range=1, multichannel=True,
                                      gaussian_weights=True) for ind in range(len(dehaze_list))]

    return ssim_list


def validation(net, val_data_loader, device, dataset_name, save_tag=False):
    psnr_list = []
    ssim_list = []
    total_time = 0
    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():

            lrs = val_data['LQs'].to(device)
            gt = val_data['GT'].to(device)

            name_gt = [i.split('/')[-1] for i in val_data['gt_name']]
            start_time = time.time()

            if color == 'raw':
                ref = val_data['ref'].to(device)
                sr = net(lrs, ref)
            else:
                sr = net(lrs)
            end_time = time.time() - start_time

            total_time += end_time
            # To calculate average PSNR
            psnr_list.extend(to_psnr(sr, gt))

            # Calculate average SSIM
            ssim_list.extend(to_ssim_skimage(sr, gt))

            # Save image
            if save_tag:
                save_image(sr.clone(), name_gt, dataset_name)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)

    # print('store:{0:.2f}, {1:.4f}'.format(sum(psnr_list[0:31]) / len(psnr_list[0:31]), sum(ssim_list[0:31]) / len(ssim_list[0:31])))
    # print('painting:{0:.2f}, {1:.4f}'.format(sum(psnr_list[31:62]) / len(psnr_list[31:62]), sum(ssim_list[31:62]) / len(ssim_list[31:62])))
    # print('train:{0:.2f}, {1:.4f}'.format(sum(psnr_list[62:93]) / len(psnr_list[62:93]), sum(ssim_list[62:93]) / len(ssim_list[62:93])))
    # print('city:{0:.2f}, {1:.4f}'.format(sum(psnr_list[93:124]) / len(psnr_list[93:124]), sum(ssim_list[93:124]) / len(ssim_list[93:124])))
    # print('tree:{0:.2f}, {1:.4f}'.format(sum(psnr_list[124:155]) / len(psnr_list[124:155]), sum(ssim_list[124:155]) / len(ssim_list[124:155])))
    # #
    # assert len(psnr_list) == 155

    print('network processing time: {}'.format(total_time))
    return avr_psnr, avr_ssim


def save_image(derain, image_name, category):
    derain_images = torch.split(derain, 1, dim=0)
    batch_num = len(derain_images)

    path = './{}_results_{}/'.format(category, scale)
    if not os.path.exists(path):
        os.mkdir(path)

    for ind in range(batch_num):
        # this function changes the image to range(0, 255)
        utils.save_image(derain_images[ind], path + '{}'.format(image_name[ind].split('.')[0] + '.png'))


def print_log(epoch, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim))
    # write training log
    path = './training_log/'
    if not os.path.exists(path):
        os.mkdir(path)

    with open(path + '{}_log.txt'.format(category), 'a+') as f:
        print(
            'Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR: {5:.2f}, Val_SSIM: {6:.4f}'
            .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim), file=f)


def adjust_learning_rate(optimizer, epoch, num_epochs, lr_decay=0.5):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    step = num_epochs // 4

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))


# EDVR

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
