import numpy
import cupy
from cupyx import cutensor
import torch

dtype = numpy.float16

# A torch.Size([4096, 768])
# B torch.Size([20, 8, 8, 12])
# B reshape torch.Size([20, 768])
# final torch.Size([4096, 20])

atorch = torch.rand((2, 4), device = 'cuda', dtype = torch.float16)
btorch = torch.rand((3, 2, 2, 1), device = 'cuda', dtype = torch.float16)

mode_a = ('a', 'b')
mode_b = ('c', 'd', 'e', 'f')
mode_c = ('a', 'c')

a = cupy.asarray(atorch)
b = cupy.asarray(btorch)
c = cupy.random.random([4096, 20]).astype(dtype)

# a = a.astype(dtype)
# b = b.astype(dtype)
# c = c.astype(dtype)

mode_a = cutensor.create_mode(*mode_a)
mode_b = cutensor.create_mode(*mode_b)
mode_c = cutensor.create_mode(*mode_c)
alpha = 1
beta = 0
print(a)
print(b)
cu = cutensor.contraction(alpha, a, mode_a, b, mode_b, beta, c, mode_c, algo = -4)
btorch = btorch.reshape(20, 768)
to = torch.tensordot(atorch, btorch, dims = ([1],[1]))

if numpy.allclose(cupy.asnumpy(cu),to.cpu().numpy(), atol=1e-1, rtol=1e-1):
    print("True")
else:
    print("False")