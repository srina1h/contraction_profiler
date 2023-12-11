import numpy
import cupy
from cupyx import cutensor
import cupyx.time
import nvtx
import torch

dtype = numpy.float32

atorch = torch.rand((32, 20), device = 'cuda')
btorch = torch.rand((20, 8, 8, 12), device = 'cuda')

# mode_a = ('a', 'b')
# mode_b = ('b', 'c', 'd')
# mode_c = ('a', 'c', 'd')
# extent = {'a': 100, 'b': 200, 'c': 100, 'd': 100}
# con_type = "ab * bcd -> acd"

# mode_a = ('a', 'b', 'c')
# mode_b = ('c', 'd', 'e')
# mode_c = ('a', 'b', 'd', 'e')
# extent = {'a': 146, 'b': 251, 'c': 187, 'd': 172, 'e': 87}
# con_type = "abc * cde -> abde"

mode_a = ('a', 'b')
mode_b = ('b', 'c', 'd', 'e')
mode_c = ('a', 'c', 'd', 'e')
extent = {'a': 32, 'b': 20, 'c': 8, 'd': 8, 'e': 12}
con_type = "ab * bcde -> acde"

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

# GETT

def con():
    with nvtx.annotate(con_type, color = "purple"):
        cutensor.contraction(alpha, a, desc_a, mode_a, b, desc_b, mode_b, beta, c, desc_c, mode_c, algo = -2)

torch.cuda.cudart().cudaProfilerStart()
perf = cupyx.time.repeat(con,n_warmup=1, n_repeat=5)
torch.cuda.cudart().cudaProfilerStop()

total_flops = 2 * numpy.prod(numpy.array(list(extent.values())))
elapsed = perf.gpu_times.mean()
print("cutensor-GETT")
print('dtype: {}'.format(numpy.dtype(dtype).name))
print(perf)
print('GFLOPS: {}'.format(total_flops / elapsed / 1e9))

# ALGO_DEFAULT

def con2():
    with nvtx.annotate(con_type, color = "purple"):
        cutensor.contraction(alpha, a, desc_a, mode_a, b, desc_b, mode_b, beta, c, desc_c, mode_c, algo = -1)

torch.cuda.cudart().cudaProfilerStart()
perf = cupyx.time.repeat(con2,n_warmup=1, n_repeat=5)
torch.cuda.cudart().cudaProfilerStop()

total_flops = 2 * numpy.prod(numpy.array(list(extent.values())))
elapsed = perf.gpu_times.mean()
print("Cutensor-ALGO_DEFAULT:")
print('dtype: {}'.format(numpy.dtype(dtype).name))
print(perf)
print('GFLOPS: {}'.format(total_flops / elapsed / 1e9))

# ALGO_TTGT

def con3():
    with nvtx.annotate(con_type, color = "purple"):
        cutensor.contraction(alpha, a, desc_a, mode_a, b, desc_b, mode_b, beta, c, desc_c, mode_c, algo = -2)

torch.cuda.cudart().cudaProfilerStart()
perf = cupyx.time.repeat(con3,n_warmup=1, n_repeat=5)
torch.cuda.cudart().cudaProfilerStop()

total_flops = 2 * numpy.prod(numpy.array(list(extent.values())))
elapsed = perf.gpu_times.mean()
print("CuTensor-ALGO_TTGT:")
print('dtype: {}'.format(numpy.dtype(dtype).name))
print(perf)
print('GFLOPS: {}'.format(total_flops / elapsed / 1e9))

# Tensordot

def con4():
    with nvtx.annotate(con_type, color = "purple"):
        torch.tensordot(a,b,[[-1],[0]])

torch.cuda.cudart().cudaProfilerStart()
perf = cupyx.time.repeat(con4,n_warmup=1, n_repeat=5)
torch.cuda.cudart().cudaProfilerStop()

total_flops = 2 * numpy.prod(numpy.array(list(extent.values())))
elapsed = perf.gpu_times.mean()

print("Tensordot:")
print('dtype: {}'.format(numpy.dtype(dtype).name))
print(perf)
print('GFLOPS: {}'.format(total_flops / elapsed / 1e9))