import torch
import torchvision
#import intel_extension_for_pytorch as ipex
import copy
import time
from utils_vis import make_dot, draw
import torch.profiler as profiler

torch.manual_seed(2020)

#model = torchvision.models.resnet50()

#model =  torchvision.models.shufflenet_v2_x2_0()

model = torchvision.models.mobilenet_v3_large()

model = model.to(memory_format=torch.channels_last).eval()


#torch._C._jit_set_texpr_fuser_enabled(False)
batch_size = 40
x = torch.randn(batch_size, 3, 224, 224).to(memory_format=torch.channels_last)

with torch.no_grad():
    traced_model = torch.jit.trace(model, x)
    #traced_model = torch.jit.script(model)
    traced_model = torch.jit.freeze(traced_model).eval()

warm_up = 200

with torch.no_grad():
    for i in range(warm_up):
        y = traced_model(x)
    #graph = traced_model.graph_for(x)
    #print(graph)
    #draw(graph).render('renet50_ipex')
    #draw(graph).render('shufflenet_v2_x2_0_ipex')
    #draw(graph).render('mobilenet_v3_small_ipex')

print("begin running...............")

num_iter = 300
fwd = 0


with torch.no_grad():
    t1  = time.time()
    for i in range(num_iter):
        y = traced_model(x)
    t2 = time.time()
    fwd = fwd + (t2 - t1)

avg_time = fwd / num_iter * 1000
print("batch_size = %d, avg time is %0.3f (ms) fps:%f"%(batch_size, avg_time, batch_size  * num_iter / fwd))

def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
    # prof.export_chrome_trace("rn50_trace_" + str(prof.step_num) + ".json")

'''
with torch.no_grad():
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=10,warmup=50,active=10),
        # son_trace_ready=torch.profiler.tensorboard_trace_handler("profiler_result")
        on_trace_ready=trace_handler) as p:
        for i in range(num_iter):
            y = traced_model(x)
            p.step()
'''
