import numpy
import cupy
from cupyx import cutensor
import cupyx.time
import nvtx
import torch

dtype = numpy.float32

# mode_a = ('a', 'b')
# mode_b = ('b', 'c', 'd')
# mode_c = ('a', 'c', 'd')
# extent = {'a': 16, 'b': 20, 'c': 16, 'd': 20}
# con_type = "ab * bcd -> acd"

mode_a = ('a', 'b', 'c')
mode_b = ('c', 'd', 'e')
mode_c = ('a', 'b', 'd', 'e')
extent = {'a': 146, 'b': 251, 'c': 187, 'd': 172, 'e': 87}
con_type = "abc * cde -> abde"

a = cupy.random.random([extent[i] for i in mode_a])
b = cupy.random.random([extent[i] for i in mode_b])
c = cupy.random.random([extent[i] for i in mode_c])
a = a.astype(dtype)
b = b.astype(dtype)
c = c.astype(dtype)

desc_a = cutensor.create_tensor_descriptor(a)
desc_b = cutensor.create_tensor_descriptor(b)
desc_c = cutensor.create_tensor_descriptor(c)

mode_a = cutensor.create_mode(*mode_a)
mode_b = cutensor.create_mode(*mode_b)
mode_c = cutensor.create_mode(*mode_c)
alpha = 1.1
beta = 0

def con():
    with nvtx.annotate(con_type, color = "purple"):
        cutensor.contraction(alpha, a, desc_a, mode_a, b, desc_b, mode_b, beta, c, desc_c, mode_c)

torch.cuda.cudart().cudaProfilerStart()
perf = cupyx.time.repeat(con,n_warmup=1, n_repeat=5)
torch.cuda.cudart().cudaProfilerStop()

total_flops = 2 * numpy.prod(numpy.array(list(extent.values())))
elapsed = perf.gpu_times.mean()

print('dtype: {}'.format(numpy.dtype(dtype).name))
print(perf)
print('GFLOPS: {}'.format(total_flops / elapsed / 1e9))