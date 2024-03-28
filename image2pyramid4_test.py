import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
if cur_dir not in sys.path:
    sys.path.append(cur_dir)
import torch
import torch.nn.functional as F
import numpy as np
import random
import time
from image2pyramid4 import avgpool_pyramid4_forward, cross_accum_forward
def max_abs(x, y):
    return torch.max(torch.abs(x-y)).cpu().item()


def test_func(func, *args, marker='torch',warm=1, rep=1):
    for i in range(warm):
        _ = func(*args)
    
    torch.cuda.synchronize()
    time_start = time.perf_counter_ns()
    for i in range(rep):
        res = func(*args)
    torch.cuda.synchronize()
    time_end = time.perf_counter_ns()
    return res, ((time_end-time_start)/1e6) / rep, marker

def torch_image2pyramid4(image:torch.Tensor):
    '''
    image: [n, 3, H, W]
    '''
    n, _, h, w = image.shape
    gray = (0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]).reshape(n, 1, h, w)

    # construct the image pyramid
    image_pyramid = []
    for l in range(4):
        size = 2 ** l
        avg1 = F.avg_pool2d(gray, kernel_size=size, stride=size)
        # import pdb; pdb.set_trace()
        image_pyramid.append(avg1)

    # compute the graident of each level, each level graident include [grad x, grad y, grad L2 norm]
    grad_pyramid = []
    kernel_x = torch.tensor([[-1, 0, 1]]).view(1, 1, 1, 3).to(dtype=torch.float32, device='cuda')
    kernel_y = torch.tensor([[-1, 0, 1]]).view(1, 1, 3, 1).to(dtype=torch.float32, device='cuda')
    for l in range(4):
        gradx = F.conv2d(image_pyramid[l], kernel_x, padding=(0, 1)) # #(1, 1, h, w)
        grady = F.conv2d(image_pyramid[l], kernel_y, padding=(1, 0))
        grad = torch.sqrt(gradx**2 + grady**2)
        grad_pyramid.append(torch.cat([gradx, grady, grad], dim=1))
    return image_pyramid, grad_pyramid

def torch_image2pyramid4_1(image:torch.Tensor):
    '''
    image: [n, 3, H, W]
    '''
    # construct the image pyramid
    image_pyramid = avgpool_pyramid4_forward(image)

    # compute the graident of each level, each level graident include [grad x, grad y, grad L2 norm]
    grad_pyramid = []
    for l in range(4):
        grad_pyramid.append(cross_accum_forward(image_pyramid[l]))
    # kernel_x = torch.tensor([[-1, 0, 1]]).view(1, 1, 1, 3).to(dtype=torch.float32, device='cuda')
    # kernel_y = torch.tensor([[-1, 0, 1]]).view(1, 1, 3, 1).to(dtype=torch.float32, device='cuda')
    # for l in range(0, 4, 1):
    #     gradx = F.conv2d(image_pyramid[l], kernel_x, padding=(0, 1)) # #(1, 1, h, w)
    #     grady = F.conv2d(image_pyramid[l], kernel_y, padding=(1, 0))
    #     grad = torch.sqrt(gradx**2 + grady**2)
    #     grad_pyramid.append(torch.cat([gradx, grady, grad], dim=1))
    return image_pyramid, grad_pyramid

if __name__ == '__main__':
    torch.cuda.deterministic = True
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    dtype = torch.float32
    dev = torch.device('cuda')
    warm = 10
    rep = 50
    
    shape_list = [
        (4, 3, 960, 528)
    ]

    for shp in shape_list:
        bsz, c, h, w = shp
        image = torch.randn((bsz, c, h, w), dtype=dtype, device=dev)
        res0, t0, mk0 = test_func(torch_image2pyramid4, image, marker='torch', warm=warm, rep=rep)
        # import pdb; pdb.set_trace()
        res1, t1, mk1 = test_func(torch_image2pyramid4_1, image, marker='cuda', warm=warm, rep=rep)
        max_diff = 0.
        image_pyramid_0, grad_pyramid_0 = res0
        image_pyramid_1, grad_pyramid_1 = res1
        # image_pyramid_0 = res0
        # image_pyramid_1 = res1
        for i in range(4):
            ip0 = image_pyramid_0[i]
            ip1 = image_pyramid_1[i]
            max_diff_tmp = max_abs(ip0, ip1)
            max_diff = max(max_diff_tmp, max_diff)
            print(f'max diff of pyramid-{i}, {max_diff_tmp}')

            gp0 = grad_pyramid_0[i]
            gp1 = grad_pyramid_1[i]
            max_diff_tmp = max_abs(gp0, gp1)
            print(f'max diff of grad-{i}, {max_diff_tmp}')
            # import pdb; pdb.set_trace()
            max_diff = max(max_diff_tmp, max_diff)
        print(f'img:{shp}, max diff = {max_diff}, time:{t0:.4f}ms -> {t1:.4f}ms')
        import pdb; pdb.set_trace()

    
    



