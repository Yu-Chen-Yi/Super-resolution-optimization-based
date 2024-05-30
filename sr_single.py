from proximal.utils.utils import *
from proximal.halide.halide import *
from proximal.lin_ops import *
from proximal.prox_fns import *
from proximal.algorithms import *
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric 
import cv2
import numpy as np
import imageio
import time

def get_kernel(gau_N=7, gau_std=1.2):
    '''
    Compute the 2D gaussian kernel

    Inputs:
        gau_N:    gaussian kernel size
        gau_std:  standard deviation of the gaussian kernel
    Outputs:
        kernel:   gaussian kernel with the shape (N, N)
    '''
    # ===== write your kernel here ===== #
    gau_N = int(gau_N) // 2 * 2 + 1  # Make gau_N an odd number
    gau = np.arange(-gau_N//2 + 1., gau_N//2 + 1.)
    k, l = np.meshgrid(gau, gau)
    Cx = -1.5
    Cy = -1.5
    kernel = np.exp(-((Cx - k)**2 + (Cy - l)**2) / (2 * gau_std**2))
    kernel = kernel / np.sum(kernel)

    return kernel

def solve(img, kernel, lamb):
    img4x_size = (4*img.shape[0], 4*img.shape[1], img.shape[2])
    img4x = cv2.resize(img, dsize=(img4x_size[1], img4x_size[0]), interpolation=cv2.INTER_CUBIC)
    tstart = time.time()
    x = Variable(img4x_size)
    # formulate problem
    prob = Problem(norm1(subsample(conv(kernel, x, dims=2), (4, 4, 1)) - img) + lamb*group_norm1(grad(x, dims=2), [3])) 
    # solve problem
    result = prob.solve(verbose=True, solver='pc', x0=img4x, max_iters=1000) 
    img_solved = x.value
    t_int = time.time() - tstart
    print('Elapsed time: {} seconds'.format(t_int))

    return img_solved 

def check_ans(img_urs_path, img_ref_path):
    img_urs = imageio.imread(img_urs_path)
    img_ref = imageio.imread(img_ref_path)
    psnr = psnr_metric(img_ref, img_urs)
    print('===> PSNR: {:.4f} dB'.format(psnr))
    ssim = ssim_metric(img_ref, img_urs, multichannel=True)
    print('===> SSIM: {:.4f} dB'.format(ssim))

if __name__ == '__main__':
    # set parameters
    gaussian_N = 11
    gaussian_std = 1.2
    lamb = 1e-2
    # read tset image
    img_test_path = './image/LR_zebra_test.png'
    img_test = imageio.imread(img_test_path)/255.0
    # solve the optimization problem
    gau_kernel = get_kernel(gau_N=gaussian_N, gau_std=gaussian_std)
    gau_rgb = np.repeat(np.expand_dims(gau_kernel, axis=2), repeats=3, axis=2)
    img_solved = solve(img_test, gau_rgb, lamb)
    # save result image
    img_out = np.round(255*np.clip(img_solved, 0.0, 1.0)).astype('uint8')
    imageio.imwrite('./result/zebra_test_single.png', img_out)
    # check answer
    img_urs_path = './result/zebra_test_single.png'
    img_ref_path =  './reference/HR_zebra_test.png'
    print('===== compare with original HR image ===== ')
    check_ans(img_urs_path, img_ref_path)
    img_ref_path =  './reference/zebra_test_single_golden.png'
    print('===== compare with TAs reference answer ===== ')
    check_ans(img_urs_path, img_ref_path)