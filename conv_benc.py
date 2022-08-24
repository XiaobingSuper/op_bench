import torch
import torch.nn.functional as F

#import intel_extension_for_pytorch as ipex

import numpy as np
import time

from timeit import Timer
torch.manual_seed(2020)

num = 200

S = [
#         [1, 1, 100, 40, 16, 3, 3, 1, 1, 1, 1],
#         [1, 2048, 4, 2, 512, 1, 1, 1, 1, 0, 0],
#         [1, 512, 4, 2, 512, 3, 3, 1, 1, 1, 1],
#         [1, 512, 4, 2, 2048, 1, 1, 1, 1, 0, 0],
#         [1, 2048, 4, 2, 512, 1, 1, 1, 1, 0, 0],
#         [1, 512, 4, 2, 512, 3, 3, 1, 1, 1, 1],
#         [1, 512, 4, 2, 2048, 1, 1, 1, 1, 0, 0],
#         [1, 2048, 4, 2, 512, 1, 1, 1, 1, 0, 0],
#         [1, 512, 4, 2, 512, 3, 3, 1, 1, 1, 1],
#         [1, 512, 4, 2, 2048, 1, 1, 1, 1, 0, 0],
#         [1, 2048, 4, 2, 512, 1, 1, 1, 1, 0, 0],
#         [1, 512, 4, 2, 512, 3, 3, 1, 1, 1, 1],
#         [1, 512, 4, 2, 2048, 1, 1, 1, 1, 0, 0],
#         [1, 2048, 4, 2, 512, 1, 1, 1, 1, 0, 0],
#         [1, 512, 4, 2, 512, 3, 3, 1, 1, 1, 1],
#         [1, 512, 4, 2, 2048, 1, 1, 1, 1, 0, 0],
#         [1, 2048, 4, 2, 512, 1, 1, 1, 1, 0, 0],
#         [1, 512, 4, 2, 512, 3, 3, 1, 1, 1, 1],
#         [1, 512, 4, 2, 2048, 1, 1, 1, 1, 0, 0],
[1, 3, 224, 224, 64, 7, 7, 2, 2, 3, 3, 1],
[1, 64, 56, 56, 128, 1, 1, 1, 1, 0, 0, 1],
[1, 128, 56, 56, 128, 3, 3, 1, 1, 1, 1, 32],
[1, 128, 56, 56, 256, 1, 1, 1, 1, 0, 0, 1],
[1, 64, 56, 56, 256, 1, 1, 1, 1, 0, 0, 1],
[1, 256, 56, 56, 128, 1, 1, 1, 1, 0, 0, 1],
[1, 128, 56, 56, 128, 3, 3, 1, 1, 1, 1, 32],
[1, 128, 56, 56, 256, 1, 1, 1, 1, 0, 0, 1],
[1, 256, 56, 56, 128, 1, 1, 1, 1, 0, 0, 1],
[1, 128, 56, 56, 128, 3, 3, 1, 1, 1, 1, 32],
[1, 128, 56, 56, 256, 1, 1, 1, 1, 0, 0, 1],
[1, 256, 56, 56, 256, 1, 1, 1, 1, 0, 0, 1],
[1, 256, 56, 56, 256, 3, 3, 2, 2, 1, 1, 32],
[1, 256, 28, 28, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 256, 56, 56, 512, 1, 1, 2, 2, 0, 0, 1],
[1, 512, 28, 28, 256, 1, 1, 1, 1, 0, 0, 1],
[1, 256, 28, 28, 256, 3, 3, 1, 1, 1, 1, 32],
[1, 256, 28, 28, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 28, 28, 256, 1, 1, 1, 1, 0, 0, 1],
[1, 256, 28, 28, 256, 3, 3, 1, 1, 1, 1, 32],
[1, 256, 28, 28, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 28, 28, 256, 1, 1, 1, 1, 0, 0, 1],
[1, 256, 28, 28, 256, 3, 3, 1, 1, 1, 1, 32],
[1, 256, 28, 28, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 28, 28, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 28, 28, 512, 3, 3, 2, 2, 1, 1, 32],
[1, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 28, 28, 1024, 1, 1, 2, 2, 0, 0, 1],
[1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 14, 14, 512, 3, 3, 1, 1, 1, 1, 32],
[1, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 14, 14, 512, 3, 3, 1, 1, 1, 1, 32],
[1, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 14, 14, 512, 3, 3, 1, 1, 1, 1, 32],
[1, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 14, 14, 512, 3, 3, 1, 1, 1, 1, 32],
[1, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 14, 14, 512, 3, 3, 1, 1, 1, 1, 32],
[1, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 14, 14, 512, 3, 3, 1, 1, 1, 1, 32],
[1, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 14, 14, 512, 3, 3, 1, 1, 1, 1, 32],
[1, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 14, 14, 512, 3, 3, 1, 1, 1, 1, 32],
[1, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 14, 14, 512, 3, 3, 1, 1, 1, 1, 32],
[1, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 14, 14, 512, 3, 3, 1, 1, 1, 1, 32],
[1, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 14, 14, 512, 3, 3, 1, 1, 1, 1, 32],
[1, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 14, 14, 512, 3, 3, 1, 1, 1, 1, 32],
[1, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 14, 14, 512, 3, 3, 1, 1, 1, 1, 32],
[1, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 14, 14, 512, 3, 3, 1, 1, 1, 1, 32],
[1, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 14, 14, 512, 3, 3, 1, 1, 1, 1, 32],
[1, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 14, 14, 512, 3, 3, 1, 1, 1, 1, 32],
[1, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 14, 14, 512, 3, 3, 1, 1, 1, 1, 32],
[1, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 14, 14, 512, 3, 3, 1, 1, 1, 1, 32],
[1, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 14, 14, 512, 3, 3, 1, 1, 1, 1, 32],
[1, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 14, 14, 512, 3, 3, 1, 1, 1, 1, 32],
[1, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 14, 14, 512, 3, 3, 1, 1, 1, 1, 32],
[1, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 0, 1],
[1, 512, 14, 14, 512, 3, 3, 1, 1, 1, 1, 32],
[1, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 1024, 3, 3, 2, 2, 1, 1, 32],
[1, 1024, 7, 7, 2048, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 14, 14, 2048, 1, 1, 2, 2, 0, 0, 1],
[1, 2048, 7, 7, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 7, 7, 1024, 3, 3, 1, 1, 1, 1, 32],
[1, 1024, 7, 7, 2048, 1, 1, 1, 1, 0, 0, 1],
[1, 2048, 7, 7, 1024, 1, 1, 1, 1, 0, 0, 1],
[1, 1024, 7, 7, 1024, 3, 3, 1, 1, 1, 1, 32],
[1, 1024, 7, 7, 2048, 1, 1, 1, 1, 0, 0, 1],
    ]
for x in range(len(S)):
#for x in range(1):
    P = S[x]
    (N, C, H, W) = P[0:4]
    #N = 40
    N = 1 
    M = P[4]
    (kernel_h, kernel_w) = P[5:7]
    (stride_h, stride_w) = P[7:9]
    (padding_h, padding_w) = P[9:11]

    X_np = np.random.randn(N, C, H, W).astype(np.float32)
    W_np = np.random.randn(M, C, kernel_h, kernel_w).astype(np.float32)
    X = torch.from_numpy(X_np).to(memory_format=torch.channels_last)
    g = P[11]
    conv2d_pt = torch.nn.Conv2d(
        C, M, (kernel_h, kernel_w), stride=(stride_h, stride_w),
        padding=(padding_h, padding_w), groups=g, bias=True)

    class ConvNet(torch.nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv2d = conv2d_pt
            self.binary = torch.add

        def forward(self, x, other):
            y = self.conv2d(x)
            result = self.binary(y, other)
            #return result
            return result.relu()

    model = ConvNet().to(memory_format=torch.channels_last).eval()
    with torch.no_grad():
        traced_model = torch.jit.script(model).eval().eval()
        traced_model = torch.jit.freeze(traced_model)

    other = model.conv2d(X).to(memory_format=torch.channels_last)

    # warm_up
    with torch.no_grad():
        for i in range(300):
            y = traced_model(X, other)

    num_iter = 300
    fwd = 0
    with torch.no_grad():
        t1  = time.time()
        for i in range(num_iter):
            y = traced_model(X, other)
        t2 = time.time()
        fwd = fwd + (t2 - t1)

    avg_time = fwd / num_iter * 1000
    print("time {}".format(avg_time))

    '''
    def pt_forward():
        with torch.no_grad():
            y = traced_model(X, other)
    #torch._C._jit_set_texpr_fuser_enabled(True)
    t = Timer("pt_forward()", "from __main__ import pt_forward, X, other") 
    t1 = t.timeit(num) / num * 1000.0
    print("time {}".format(t1))
    '''
