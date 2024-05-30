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

def get_kernel(gau_N=7, gau_std=1.2, mv_x=0, mv_y=0):
    '''
    Compute the 2D gaussian kernel

    Inputs:
        gau_N:   gaussian kernel size
        gau_std: standard deviation of the gaussian kernel
        mv_x:    motion shift of the input image along x-axis
        mv_y:    motion shift of the input image along y-axis  
    Outputs:
        kernel:  gaussian kernel with the shape (N, N)
    '''
    # ===== write your kernel here ===== #
    gau_N = int(gau_N) // 2 * 2 + 1  # Make gau_N an odd number
    gau = np.arange(-gau_N//2 + 1., gau_N//2 + 1.)
    k, l = np.meshgrid(gau, gau)
    Cx = -1.5 + 4*mv_x
    Cy = -1.5 + 4*mv_y
    kernel = np.exp(-((Cx - k)**2 + (Cy - l)**2) / (2 * gau_std**2))
    kernel = kernel / np.sum(kernel)

    return kernel

def solve(imgs, filters, lamb, N):
    '''
    Solve the optimization problem by proximal 

    Inputs:
        imgs:       eight low-resolution images, each image i corresponds to imgs[i], for i in the range [0, 7] 
        filters:    eight gaussian filters, each filter i corresponds to filter[i], for i in the range [0, 7] 
        lamb:       regularization term cofficient
    Outputs: 
        img_solved: solved image
    ''' 

    img4x_size = (4*imgs[0].shape[0], 4*imgs[0].shape[1], imgs[0].shape[2]) 
    img4x = cv2.resize(imgs[0], dsize=(img4x_size[1], img4x_size[0]), interpolation=cv2.INTER_CUBIC)
    tstart = time.time()
    x = Variable(img4x_size)
    # ===== formulate the problem here, you can refer to sr_single.py ===== #
    #prob = Problem(norm1(subsample(conv(filters, x, dims=2), (4, 4, 1)) - imgs) + lamb*group_norm1(grad(x, dims=2), [3])) 
    terms = []
    for i in range(N):
        term = norm1(subsample(conv(filters[i], x, dims=2), (4, 4, 1)) - imgs[i])
        terms.append(term)
    data_term = np.sum(terms)
    reg_term = lamb * group_norm1(grad(x, dims=2), [3])
    prob = Problem(data_term + reg_term)
    
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

def Experiment(N):
    # set parameters
    gaussian_N = 13
    gaussian_std = 1.2
    lamb = 1e-2
    # read test images & load kernels
    img_test_path = './image/LR_zebra_test_mvx{:.2f}_mvy{:.2f}.png'
    _motion_shifts = [(0, 0), (0.22, 0.34), (-0.31, 0.18), (0.25, -0.29), (-0.36, -0.21), (-0.1, -0.1), (-0.12, 0.11), (0.2, 0.3)]
    motion_shifts = _motion_shifts[0:N]
    img_tests = []
    gau_rgbs = []
    for motion_shift in motion_shifts:
        print(motion_shift)
        img_tests.append(imageio.imread(img_test_path.format(*motion_shift))/255.0)
        gau_kernel = get_kernel(gaussian_N, gaussian_std, *motion_shift)
        gau_rgbs.append(np.repeat(np.expand_dims(gau_kernel, axis=2), repeats=3, axis=2))
    # solve the optimization problem
    img_solved = solve(img_tests, gau_rgbs, lamb, N)
    # save result image
    img_out = np.round(255*np.clip(img_solved, 0.0, 1.0)).astype('uint8')
    imageio.imwrite(f'./result/zebra_test_{N}.png', img_out)
    # check answer
    img_urs_path = f'./result/zebra_test_{N}.png'
    img_ref_path =  './reference/HR_zebra_test.png'
    print('===== compare with original HR image ===== ')
    check_ans(img_urs_path, img_ref_path)
    img_ref_path =  './reference/zebra_test_eight_golden.png'
    print('===== compare with TAs reference answer ===== ')
    check_ans(img_urs_path, img_ref_path)

if __name__ == '__main__':
    for i in range(2,9):
        Experiment(i)