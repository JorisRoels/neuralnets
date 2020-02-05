import math

import kornia
import torch


def comb(n, r):
    """
    Compute the binomial coefficient of n over r
    :param n: total number of objects
    :param r: number of selected objects
    :return: binomial coefficient of n over r
    """

    num = n
    den = 1

    for i in range(2, r + 1):
        num *= (n - i + 1)
        den *= i

    return num / den


def bilateral_filter(x, s_r=15, s_s=100, T=255):
    """
    Fast implementation of the bilateral filter using the technique described in
        K. N. Chaudhury, D. Sage, and M. Unser,
        "Fast O(1) bilateral filtering using trigonometric range kernels",
        IEEE Transactions on Image Processing
    :param x: (N, C, H, W) tensor of image data
    :param s_r: intensity range damping parameter
    :param s_s: spatial damping parameter
    :param T: dynamic range
    :return: (N, C, H, W) tensor of filtered image data
    """

    return extended_bilateral_filter(x, x, s_r=s_r, s_s=s_s, T=T)


def extended_bilateral_filter(x, p, s_r=15, s_s=100, T=255):
    """
    Fast implementation of the bilateral filter using the technique described in
        K. N. Chaudhury, D. Sage, and M. Unser,
        "Fast O(1) bilateral filtering using trigonometric range kernels",
        IEEE Transactions on Image Processing
    :param x: (N, C, H, W) tensor of intensity data to compute weights on
    :param p: (N, C, H, W) tensor of image data to be filtered
    :param s_r: intensity range damping parameter
    :param s_s: spatial damping parameter
    :param T: dynamic range
    :return: (N, C, H, W) tensor of filtered image data
    """

    Nmax = 100

    gamma = math.pi / (2 * T)
    rho = gamma * s_r
    if s_r > 1 / gamma ** 2:
        N = Nmax
    else:
        N = int(1 / (gamma * s_r) ** 2)

    N = min(N, Nmax)

    gauss = kornia.filters.GaussianBlur((9, 9), (s_s, s_s))

    v = torch.mul(x, gamma / (rho * math.sqrt(N)))
    num = torch.zeros_like(x)
    den = torch.zeros_like(x)
    for n in range(N + 1):
        h = torch.cos(torch.mul(v, 2 * n - N))
        g = torch.mul(p, h)
        d = torch.mul(h, (1 / 2 ** N) * comb(N, n))
        hh = gauss(h)
        gg = gauss(g)
        num += torch.mul(d, gg)
        den += torch.mul(d, hh)

    return torch.div(num, den)
