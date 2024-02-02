import numpy
import cupy
from cupyx import cutensor
import cupyx.time
import nvtx
import torch

dtype = numpy.float32

# A torch.Size([4096, 768])
# B torch.Size([20, 8, 8, 12])
# B reshape torch.Size([20, 768])
# final torch.Size([4096, 20])

atorch = torch.rand((4096, 768), device = 'cuda')
btorch = torch.rand((20, 768), device = 'cuda')

mode_a = ('a', 'b')
mode_b = ('c', 'd', 'e', 'f')
mode_c = ('a', 'c')
extent = {'a': 4096, 'b': 768, 'c': 20, 'd': 8, 'e': 8, 'f': 12}
con_type = "ab * cdef -> ac"

# atorch = torch.rand((4096, 768), device = 'cuda')
# btorch = torch.rand((20, 768), device = 'cuda')

# mode_a = ('a', 'b')
# mode_b = ('c', 'b')
# mode_c = ('a', 'c')
# extent = {'a': 4096, 'b': 768, 'c': 20}
# con_type = "ab * cb -> ac"

# mode_a = ('a', 'b', 'c')
# mode_b = ('c', 'd', 'e')
# mode_c = ('a', 'b', 'd', 'e')
# extent = {'a': 146, 'b': 251, 'c': 187, 'd': 172, 'e': 87}
# con_type = "abc * cde -> abde"

# mode_a = ('a', 'b')
# mode_b = ('b', 'c', 'd', 'e')
# mode_c = ('a', 'c', 'd', 'e')
# extent = {'a': 4096, 'b': 20, 'c': 12, 'd': 16, 'e': 16}
# con_type = "ab * bcde -> acde"

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
perf1 = cupyx.time.repeat(con,n_warmup=1, n_repeat=5)
torch.cuda.cudart().cudaProfilerStop()

total_flops = 2 * numpy.prod(numpy.array(list(extent.values())))
elapsed = perf1.gpu_times.mean()
print("cutensor-GETT")
print('dtype: {}'.format(numpy.dtype(dtype).name))
print(perf1)
print('Avg CPU time: {}'.format(perf1.cpu_times.mean()))
print('Avg Gpu time: {}'.format(perf1.gpu_times.mean()))
print('Total avg time: {}'.format(perf1.cpu_times.mean() + perf1.gpu_times.mean()))

# ALGO_DEFAULT

def con2():
    with nvtx.annotate(con_type, color = "purple"):
        cutensor.contraction(alpha, a, desc_a, mode_a, b, desc_b, mode_b, beta, c, desc_c, mode_c, algo = -1)

torch.cuda.cudart().cudaProfilerStart()
perf2 = cupyx.time.repeat(con2,n_warmup=1, n_repeat=5)
torch.cuda.cudart().cudaProfilerStop()

total_flops = 2 * numpy.prod(numpy.array(list(extent.values())))
elapsed = perf2.gpu_times.mean()
print("Cutensor-ALGO_DEFAULT:")
print('dtype: {}'.format(numpy.dtype(dtype).name))
print(perf2)
print('Avg CPU time: {}'.format(perf2.cpu_times.mean()))
print('Avg Gpu time: {}'.format(perf2.gpu_times.mean()))
print('Total avg time: {}'.format(perf2.cpu_times.mean() + perf2.gpu_times.mean()))

# ALGO_TTGT

def con3():
    with nvtx.annotate(con_type, color = "purple"):
        cutensor.contraction(alpha, a, desc_a, mode_a, b, desc_b, mode_b, beta, c, desc_c, mode_c, algo = -2)

torch.cuda.cudart().cudaProfilerStart()
perf3 = cupyx.time.repeat(con3,n_warmup=1, n_repeat=5)
torch.cuda.cudart().cudaProfilerStop()

total_flops = 2 * numpy.prod(numpy.array(list(extent.values())))
elapsed = perf3.gpu_times.mean()
print("CuTensor-ALGO_TTGT:")
print('dtype: {}'.format(numpy.dtype(dtype).name))
print(perf3)
print('Avg CPU time: {}'.format(perf3.cpu_times.mean()))
print('Avg Gpu time: {}'.format(perf3.gpu_times.mean()))
print('Total avg time: {}'.format(perf3.cpu_times.mean() + perf3.gpu_times.mean()))

# Tensordot

def con4():
    with nvtx.annotate(con_type, color = "purple"):
        torch.tensordot(atorch, btorch, dims = ([1],[1]))

torch.cuda.cudart().cudaProfilerStart()
perf4 = cupyx.time.repeat(con4,n_warmup=1, n_repeat=5)
torch.cuda.cudart().cudaProfilerStop()

total_flops = 2 * numpy.prod(numpy.array(list(extent.values())))
elapsed = perf4.gpu_times.mean()

print("Tensordot:")
print('dtype: {}'.format(numpy.dtype(dtype).name))
print(perf4)
print('Avg CPU time: {}'.format(perf4.cpu_times.mean()))
print('Avg Gpu time: {}'.format(perf4.gpu_times.mean()))
print('Total avg time: {}'.format(perf4.cpu_times.mean() + perf4.gpu_times.mean()))

# Correctness check

cu = cutensor.contraction(alpha, a, desc_a, mode_a, b, desc_b, mode_b, beta, c, desc_c, mode_c, algo = -2)

to = torch.tensordot(atorch, btorch, dims = ([1],[1]))

if cu.shape == to.shape:
    print("Shapes are equal")
else:
    print("Shapes are not equal")

if cupy.asnumpy(cu) == to.numpy:
    print("Results are equal")
else:
    print("Results are not equal")
    print(cupy.asnumpy(cu))
    print("-----------------")
    print(to.numpy)