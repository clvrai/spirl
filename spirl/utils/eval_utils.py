from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from spirl.utils.general_utils import get_dim_inds
from spirl.utils.pytorch_utils import ten2ar, ar2ten
from torch.autograd import Variable


class metric:
    """ A function decorator that adds an argument 'per_datum'. """

    def __init__(self, func):
        self.func = func

    def __call__(self, estimates, targets, per_datum=False):
        """

        :param estimates:
        :param targets:
        :param per_datum: If this is True, return a tensor of shape: [batch_size], otherwise: [1]
        :return:
        """
        error = self.func(estimates, targets)
        if isinstance(error, torch.Tensor): error = ten2ar(error)
        if per_datum:
            return np.mean(error, axis=get_dim_inds(error)[1:])
        else:
            return np.mean(error)


@metric
def ssim(img1, img2, window_size=11):
    if isinstance(img1, np.ndarray): img1 = ar2ten(img1)
    if isinstance(img2, np.ndarray): img2 = ar2ten(img2)

    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel).detach().cpu().numpy()


@metric
def psnr(estimates, targets, data_dims=3):
    # NOTE: PSNR is not dimension-independent. The number of dimensions which are part of the metric has to be specified
    # I.e 2 for grayscale, 3 for color images.
    if isinstance(estimates, torch.Tensor): estimates = ten2ar(estimates)
    if isinstance(targets, torch.Tensor): targets = ten2ar(targets)

    estimates = (estimates + 1) / 2
    targets = (targets + 1)/2

    max_pix_val = 1.0
    tolerance = 0.001
    assert (0 - tolerance) <= np.min(targets) and np.max(targets) <= max_pix_val * (1 + tolerance)
    assert (0 - tolerance) <= np.min(estimates) and np.max(estimates) <= max_pix_val * (1 + tolerance)

    mse = (np.square(estimates - targets))
    mse = np.mean(mse, axis=get_dim_inds(mse)[-data_dims:])

    psnr = 10 * np.log(max_pix_val / mse) / np.log(10)
    if np.any(np.isinf(psnr)):
        import pdb; pdb.set_trace()
    return psnr


@metric
def mse(estimates, targets):
    if isinstance(estimates, torch.Tensor): estimates = ten2ar(estimates)
    if isinstance(targets, torch.Tensor): targets = ten2ar(targets)
    return np.square(estimates - targets)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map

