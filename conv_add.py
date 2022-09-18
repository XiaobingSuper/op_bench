import torch
import torch.nn.functional as F
import torch.profiler as profiler
#import intel_extension_for_pytorch as ipex
import copy

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
    P = S[-2]
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
    conv2d_1 = torch.nn.Conv2d(
        C, M, (kernel_h, kernel_w), stride=(stride_h, stride_w),
        padding=(padding_h, padding_w), groups=g, bias=True)

    conv2d_2 = torch.nn.Conv2d(
        C, M, (kernel_h, kernel_w), stride=(stride_h, stride_w),
        padding=(padding_h, padding_w), groups=g, bias=True)


    class ConvNet(torch.nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv1 = conv2d_1
            self.binary = torch.add
            self.conv2 = conv2d_2

        def forward(self, x):
            y1 = self.conv1(x)
            y2 = self.conv2(x)
            result = self.binary(y1, y2)
            return result
            #return result.relu()

    model = ConvNet().to(memory_format=torch.channels_last).eval()
    with torch.no_grad():
        traced_model = torch.jit.script(model).eval().eval()
        traced_model = torch.jit.freeze(traced_model)

    #other = model.conv2d(X).to(memory_format=torch.channels_last)

    # warm_up
    with torch.no_grad():
        for i in range(300):
            y = traced_model(X)
        #print(traced_model.graph_for(X))

    #print("begin running.............")
    num_iter = 300
    fwd = 0
    with torch.no_grad():
        t1  = time.time()
        for i in range(num_iter):
            y = traced_model(X)
        t2 = time.time()
        fwd = fwd + (t2 - t1)

    avg_time = fwd / num_iter * 1000
    print("time {}".format(avg_time))
    def trace_handler(prof):
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))

    '''
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=10,warmup=50,active=10),
        # son_trace_ready=torch.profiler.tensorboard_trace_handler("profiler_result")
        on_trace_ready=trace_handler) as p:
        with torch.no_grad():
            for i in range(300):
                y = traced_model(X)
                p.step()
    '''
