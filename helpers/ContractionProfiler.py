import numpy
import cupy
from cupyx import cutensor
import cupyx.time
import nvtx
import torch
from helpers.Dimensions import *

algorithms = ["ALGO_DEFAULT","ALGO_TTGT", "ALGO_TGETT", "ALGO_GETT", "ALGO_DEFAULT_PATIENT" , "tensordot"]

# ALGO_DEFAULT_PATIENT = -6  # NOQA, Uses the more accurate but also more time-consuming performance model
# ALGO_GETT = -4             # NOQA, Choose the GETT algorithm
# ALGO_TGETT = -3            # NOQA, Transpose (A or B) + GETT
# ALGO_TTGT = -2             # NOQA, Transpose-Transpose-GEMM-Transpose (requires additional memory)
# ALGO_DEFAULT = -1          # NOQA, Lets the internal heuristic choose

class ContractionProfiler:
    def __init__(self, dimensions: Dimensions, contractionLabel: str = "") -> None:
        self.dimensions = dimensions
        self.setDtype(self.dimensions.dataType)
        self.set_modes(self.dimensions.con_type)
        self.extent = self.set_extents(self.dimensions.adim, self.dimensions.bdim, self.dimensions.cdim, self.mode_a, self.mode_b, self.mode_c)

        self.atorch = torch.rand(self.dimensions.adim, device = 'cuda', dtype = self.torchdType)
        self.btorch = torch.rand(self.dimensions.bdim, device = 'cuda', dtype = self.torchdType)

        self.a = cupy.random.random([self.extent[i] for i in self.mode_a])
        self.b = cupy.random.random([self.extent[i] for i in self.mode_b])
        self.c = cupy.random.random([self.extent[i] for i in self.mode_c])
        print(self.dtype)
        self.a = self.a.astype(self.dtype)
        self.b = self.b.astype(self.dtype)
        self.c = self.c.astype(self.dtype)

        self.desc_a = cutensor.create_tensor_descriptor(self.a)
        self.desc_b = cutensor.create_tensor_descriptor(self.b)
        self.desc_c = cutensor.create_tensor_descriptor(self.c)

        self.mode_a = cutensor.create_mode(*self.mode_a)
        self.mode_b = cutensor.create_mode(*self.mode_b)
        self.mode_c = cutensor.create_mode(*self.mode_c)
        self.alpha = 1
        self.beta = 0

        self.contractionLabel = contractionLabel

    def setDtype(self, dataType) -> None:
        if dataType == "float32":
            self.dtype = numpy.float32
            self.torchdType = torch.float32
        elif dataType == "float16":
            self.dtype = numpy.float16
            self.torchdType = torch.float16

    def get_cutensor_algo(self, algo_number) -> str:
        if algo_number == -6:
            return "ALGO_DEFAULT_PATIENT"
        elif algo_number == -4:
            return "ALGO_GETT"
        elif algo_number == -3:
            return "ALGO_TGETT"
        elif algo_number == -2:
            return "ALGO_TTGT"
        elif algo_number == -1:
            return "ALGO_DEFAULT"
    
    def set_modes(self, con_type) -> None:
        AB = con_type.split("->")[0]
        A = AB.split("*")[0]
        B = AB.split("*")[1]
        C = con_type.split("->")[1]

        self.mode_a = tuple([i for i in A.split()[0]])
        self.mode_b = tuple([j for j in B.split()[0]])
        self.mode_c = tuple([k for k in C.split()[0]])
    
    def set_extents(self, adim, bdim, cdim, mode_a, mode_b, mode_c) -> dict:
        extent_a = {}
        extent_b = {}
        extent_c = {}

        def populate_extent(extent, mode, dim):
            # print(extent, mode, dim, len(mode), len(dim))
            for i in range(len(mode)):
                extent[mode[i]] = dim[i]
            return extent
        
        n_extent_a = populate_extent(extent_a, mode_a, adim)
        n_extent_b = populate_extent(extent_b, mode_b, bdim)
        n_extent_c = populate_extent(extent_c, mode_c, cdim)
        # print(n_extent_a, n_extent_b, n_extent_c)
        # print(n_extent_a | n_extent_b | n_extent_c)
        return n_extent_a | n_extent_b | n_extent_c
    
    def profile_cutensor(self, algo_number) -> list:
        def con():
            with nvtx.annotate(self.dimensions.con_type + self.get_cutensor_algo(algo_number) + self.contractionLabel, color = "purple"):
                cutensor.contraction(self.alpha, self.a, self.mode_a, self.b, self.mode_b, self.beta, self.c, self.mode_c, algo = algo_number)

        torch.cuda.cudart().cudaProfilerStart()
        perf = cupyx.time.repeat(con,n_warmup=1, n_repeat=5)
        torch.cuda.cudart().cudaProfilerStop()

        return [perf.cpu_times.mean(), perf.gpu_times.mean()]
    
    def profile_tensordot(self) -> list:
        def con():
            with nvtx.annotate(self.dimensions.con_type + "tdot" + self.contractionLabel, color = "purple"):
                torch.tensordot(self.atorch, self.btorch, dims = self.dimensions.tdotConDim)

        torch.cuda.cudart().cudaProfilerStart()
        perf = cupyx.time.repeat(con,n_warmup=1, n_repeat=5)
        torch.cuda.cudart().cudaProfilerStop()

        return [perf.cpu_times.mean(), perf.gpu_times.mean()]

    def check_correctness(self, algo_number) -> bool:
        cu = cutensor.contraction(self.alpha, self.a, self.mode_a, self.b, self.mode_b, self.beta, self.c, self.mode_c, algo = algo_number)
        to = torch.tensordot(self.atorch, self.btorch, dims = self.dimensions.tdotConDim)
        print(cupy.asnumpy(cu))
        print(to.numpy)

        if numpy.array_equal(cupy.asnumpy(cu), to.numpy):
            return True
        else:
            return False
    
    def profile_all(self) -> (list, list, list, list, list, list, bool):
        cutensor_default = self.profile_cutensor(-1)
        cutensor_ttgt = self.profile_cutensor(-2)
        cutensor_tgett = self.profile_cutensor(-3)
        cutensor_gett = self.profile_cutensor(-4)
        cutensor_default_patient = self.profile_cutensor(-6)
        tensordot = self.profile_tensordot()

        correctness = self.check_correctness(-4)

        lowest_CPU = self.fastest_time([cutensor_default[0], cutensor_ttgt[0], cutensor_tgett[0], cutensor_gett[0], cutensor_default_patient[0], tensordot[0]])
        lowest_GPU = self.fastest_time([cutensor_default[1], cutensor_ttgt[1], cutensor_tgett[1], cutensor_gett[1], cutensor_default_patient[1], tensordot[1]])

        return [self.contractionLabel, cutensor_default, cutensor_ttgt, cutensor_tgett, cutensor_gett, cutensor_default_patient, tensordot, correctness, lowest_CPU, lowest_GPU]

    def fastest_time(self, inp) -> int:
        return algorithms[inp.index(min(inp))]