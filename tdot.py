import numpy
import cupyx.time
import nvtx
import torch

dtype = numpy.float32
a = torch.rand((234, 357, 265), device = 'cuda')
b = torch.rand((265, 187, 276), device = 'cuda')
# con_type = "ab * bcd -> acd"
con_type = "abc * cde -> abde"
extent = {'a': 234, 'b': 357, 'c': 265, 'd': 187, 'e': 276}

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