import numpy
import cupyx.time
import nvtx
import torch

dtype = numpy.float32
a = torch.rand((16, 16, 20), device = 'cuda')
b = torch.rand((20, 12, 20), device = 'cuda')
# con_type = "ab * bcd -> acd"
con_type = "abc * cde -> abde"
extent = {'a': 16, 'b': 16, 'c': 20, 'd': 12, 'e': 20}

def con():
    with nvtx.annotate(con_type, color = "purple"):
        torch.tensordot(a,b,[[-1],[0]])

torch.cuda.cudart().cudaProfilerStart()
perf = cupyx.time.repeat(con,n_warmup=1, n_repeat=5)
torch.cuda.cudart().cudaProfilerStop()

total_flops = 2 * numpy.prod(numpy.array(list(extent.values())))
elapsed = perf.gpu_times.mean()

print('dtype: {}'.format(numpy.dtype(dtype).name))
print(perf)
print('GFLOPS: {}'.format(total_flops / elapsed / 1e9))