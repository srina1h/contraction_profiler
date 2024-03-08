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

atorch = torch.rand((4096, 768), device = 'cuda', dtype = torch.float32)
btorch = torch.rand((20, 768), device = 'cuda', dtype = torch.float32)

mode_a = ('a', 'b')
mode_b = ('c', 'b')
mode_c = ('a', 'c')
extent = {'a': 4096, 'b': 768, 'c': 20}
con_type = "ab * cb -> ac"

# atorch = torch.rand((4096, 768), device = 'cuda')
# btorch = torch.rand((20, 768), device = 'cuda')

# mode_a = ('a', 'b')
# mode_b = ('b', 'c')
# mode_c = ('a', 'c')
# extent = {'a': 2, 'b': 2, 'c': 2}
# con_type = "ab * cb -> ac"
# atorch = torch.rand((2,2), device = 'cuda')
# btorch = torch.rand((2,2), device = 'cuda')

# mode_a = ('a', 'b', 'c')
# mode_b = ('c', 'd', 'e')
# mode_c = ('a', 'b', 'd', 'e')
# extent = {'a': 12, 'b': 8, 'c': 20, 'd': 8, 'e': 20}
# con_type = "abc * cde -> abde"
# atorch = torch.rand((12, 8, 20), device = 'cuda', dtype = torch.float16)
# btorch = torch.rand((20, 8, 20), device = 'cuda', dtype = torch.float16)

# mode_a = ('a', 'b')
# mode_b = ('b', 'c', 'd', 'e')
# mode_c = ('a', 'c', 'd', 'e')
# extent = {'a': 4096, 'b': 20, 'c': 12, 'd': 16, 'e': 16}
# con_type = "ab * bcde -> acde"

a = cupy.random.random([extent[i] for i in mode_a])
b = cupy.random.random([extent[i] for i in mode_b])
c = cupy.random.random([extent[i] for i in mode_c])
# atorch = torch.as_tensor(a, device = 'cuda')
# btorch = torch.as_tensor(b, device = 'cuda')
a = a.astype(dtype)
b = b.astype(dtype)
c = c.astype(dtype)

# atorch = torch.from_numpy(cupy.asnumpy(a)).to('cuda')
# btorch = torch.from_numpy(cupy.asnumpy(b)).to('cuda')

mode_a = cutensor.create_mode(*mode_a)
mode_b = cutensor.create_mode(*mode_b)
mode_c = cutensor.create_mode(*mode_c)
alpha = 1
beta = 0

# GETT

def con():
    with nvtx.annotate(con_type + "gett", color = "purple"):
        cutensor.contraction(alpha, a, mode_a, b, mode_b, beta, c, mode_c, algo = -4)

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
    with nvtx.annotate(con_type + "def", color = "purple"):
        cutensor.contraction(alpha, a, mode_a, b, mode_b, beta, c, mode_c, algo = -1)

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
    with nvtx.annotate(con_type + "ttgt", color = "purple"):
        cutensor.contraction(alpha, a, mode_a, b, mode_b, beta, c, mode_c, algo = -2)

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
    with nvtx.annotate(con_type + "tdot", color = "purple"):
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

def con5():
    with nvtx.annotate(con_type + "esum", color = "purple"):
        cupy.einsum('ab, bc->ac', a, b)

torch.cuda.cudart().cudaProfilerStart()
perf5 = cupyx.time.repeat(con5,n_warmup=1, n_repeat=5)
torch.cuda.cudart().cudaProfilerStop()

total_flops = 2 * numpy.prod(numpy.array(list(extent.values())))
elapsed = perf5.gpu_times.mean()

print("Einsum:")
print('dtype: {}'.format(numpy.dtype(dtype).name))
print(perf5)
print('Avg CPU time: {}'.format(perf5.cpu_times.mean()))
print('Avg Gpu time: {}'.format(perf5.gpu_times.mean()))
print('Total avg time: {}'.format(perf5.cpu_times.mean() + perf5.gpu_times.mean()))

# Correctness check

cu = cutensor.contraction(alpha, a, mode_a, b, mode_b, beta, c, mode_c, algo = -2)

to = torch.tensordot(atorch, btorch, dims = ([1],[1]))

cup = cupy.einsum('ab, bc->ac', a, b)

if cu.shape == to.shape:
    print("Shapes are equal")
else:
    print("Shapes are not equal")
    print(cu.shape)
    print(to.shape)
    print(cup.shape)

if numpy.array_equal(cupy.asnumpy(cup), to.numpy):
    print("Results are equal einsum tdot")
else:
    print("Results are not equal einsum tdot")
    # print(cupy.asnumpy(cup))
    # print("-----------------")
    # print(to.cpu().numpy())

if numpy.array_equal(cupy.asnumpy(cup), cupy.asnumpy(cu)):
    print("Results are equal einsum cutensor")
else:
    print("Results are not equal einsum cutensor")
    # print(cupy.asnumpy(cup))
    # print("-----------------")
    # print(cupy.asnumpy(cu))

if numpy.array_equal(cupy.asnumpy(cu), to.numpy):
    print("Results are equal")
else:
    print("Results are not equal")
    # print(cupy.asnumpy(cu))
    # print("-----------------")
    # print(to.cpu().numpy())
print(atorch)
print(btorch)
print(a)
print(b)
print("-----------------")
print(cupy.asnumpy(cup))
print("-----------------")
print(cupy.asnumpy(cu))
print("-----------------")
print(to.cpu().numpy())