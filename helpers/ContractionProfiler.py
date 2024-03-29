import numpy
import cupy
from cupyx import cutensor
import cupyx.time
import nvtx
import torch
import platform
# from cuquantum import contract
from helpers.Dimensions import *

algorithms = ["ALGO_DEFAULT","ALGO_TTGT", "ALGO_TGETT", "ALGO_GETT", "ALGO_DEFAULT_PATIENT" , "cuquantum", "tensordot", "einsum"]

# ALGO_DEFAULT_PATIENT = -6  # NOQA, Uses the more accurate but also more time-consuming performance model
# ALGO_GETT = -4             # NOQA, Choose the GETT algorithm
# ALGO_TGETT = -3            # NOQA, Transpose (A or B) + GETT
# ALGO_TTGT = -2             # NOQA, Transpose-Transpose-GEMM-Transpose (requires additional memory)
# ALGO_DEFAULT = -1          # NOQA, Lets the internal heuristic choose

class ContractionProfiler:
    def __init__(self, dimensions: Dimensions, contractionLabel: str = "") -> None:
        self.hasCrashed = False
        self.dimensions = dimensions
        self.setDtype(self.dimensions.dataType)
        self.set_modes(self.dimensions.con_type)
        self.extent = self.set_extents(self.dimensions.adim, self.dimensions.bdim, self.dimensions.cdim, self.mode_a, self.mode_b, self.mode_c)
        self.contractionLabel = contractionLabel

        try:
            self.a = cupy.random.random([self.extent[i] for i in self.mode_a])
            self.b = cupy.random.random([self.extent[i] for i in self.mode_b])
            self.c = cupy.random.random([self.extent[i] for i in self.mode_c])
        except:
            print("Memory allocation error")
            self.hasCrashed = True
        else:
            self.a = self.a.astype(self.dtype)
            self.b = self.b.astype(self.dtype)
            self.c = self.c.astype(self.dtype)

            self.atorch = torch.as_tensor(self.a, device = 'cuda')
            self.btorch = torch.as_tensor(self.b, device = 'cuda')

            self.mode_a = cutensor.create_mode(*self.mode_a)
            self.mode_b = cutensor.create_mode(*self.mode_b)
            self.mode_c = cutensor.create_mode(*self.mode_c)
            self.alpha = 1
            self.beta = 0

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
        self.cqinp  = A + ',' + B
        C = con_type.split("->")[1]

        self.mode_a = tuple([i for i in A.split()[0]])
        self.mode_b = tuple([j for j in B.split()[0]])
        self.mode_c = tuple([k for k in C.split()[0]])
    
    def set_extents(self, adim, bdim, cdim, mode_a, mode_b, mode_c) -> dict:
        extent_a = {}
        extent_b = {}
        extent_c = {}

        def populate_extent(extent, mode, dim):
            for i in range(len(mode)):
                extent[mode[i]] = dim[i]
            return extent
        
        n_extent_a = populate_extent(extent_a, mode_a, adim)
        n_extent_b = populate_extent(extent_b, mode_b, bdim)
        n_extent_c = populate_extent(extent_c, mode_c, cdim)
        if platform.version() < '3.9':
            temp = {**n_extent_a, **n_extent_b}
            return {**temp, **n_extent_c}
        else:
            return n_extent_a | n_extent_b | n_extent_c
    
    def profile_cutensor(self, algo_number) -> list:
        def con():
            with nvtx.annotate(self.dimensions.con_type + self.get_cutensor_algo(algo_number) + self.contractionLabel, color = "purple"):
                cutensor.contraction(self.alpha, self.a, self.mode_a, self.b, self.mode_b, self.beta, self.c, self.mode_c, algo = algo_number)

        torch.cuda.cudart().cudaProfilerStart()
        try:
            perf = cupyx.time.repeat(con,n_warmup=1, n_repeat=5)
        except RuntimeError as e:
            print(str(e) + " - CuTensor Err (CUDA ERROR: SEGMENT not initialized usually due to OOM)")
            return [float('inf'), float('inf')]
        except:
            print("Error in cutensor")
            return [float('inf'), float('inf')]
        torch.cuda.cudart().cudaProfilerStop()

        return [perf.cpu_times.mean(), perf.gpu_times.mean()]

    def profile_cuquantum(self) -> list:
        def con():
            with nvtx.annotate(self.dimensions.con_type + "cuq" + self.contractionLabel, color = "purple"):
                contract(self.cqinp, self.atorch, self.btorch)

        torch.cuda.cudart().cudaProfilerStart()
        try:
            perf = cupyx.time.repeat(con,n_warmup=1, n_repeat=5)
        except RuntimeError as e:
            print(str(e) + " - CuQuantum Err (CUDA ERROR: SEGMENT not initialized usually due to OOM)")
            return [float('inf'), float('inf')]
        except:
            print("Error in cuQuantum")
            return [float('inf'), float('inf')]
        torch.cuda.cudart().cudaProfilerStop()

        return [perf.cpu_times.mean(), perf.gpu_times.mean()]
    
    def profile_tensordot(self) -> list:
        def con():
            with nvtx.annotate(self.dimensions.con_type + "tdot" + self.contractionLabel, color = "purple"):
                torch.tensordot(self.atorch, self.btorch, dims = self.dimensions.tdotConDim)

        torch.cuda.cudart().cudaProfilerStart()
        try:
            perf = cupyx.time.repeat(con,n_warmup=1, n_repeat=5)
        except RuntimeError as e:
            print(str(e) + " - Tensordot Err (CUDA ERROR: SEGMENT not initialized usually due to OOM)")
            return [float('inf'), float('inf')]
        except:
            print("Error in tensordot")
            return [float('inf'), float('inf')]
        torch.cuda.cudart().cudaProfilerStop()

        return [perf.cpu_times.mean(), perf.gpu_times.mean()]
    
    def profile_einsum(self, con_type) -> list:
        def con():
            with nvtx.annotate(con_type + "ein" + self.contractionLabel, color = "purple"):
                cupy.einsum(self.parse_contype_einsum(con_type), self.a, self.b)

        torch.cuda.cudart().cudaProfilerStart()
        try:
            perf = cupyx.time.repeat(con,n_warmup=1, n_repeat=5)
        except RuntimeError as e:
            print(str(e) + " - Einsum Err (CUDA ERROR: SEGMENT not initialized usually due to OOM)")
            return [float('inf'), float('inf')]
        except:
            print("Error in einsum")
            return [float('inf'), float('inf')]
        torch.cuda.cudart().cudaProfilerStop()

        return [perf.cpu_times.mean(), perf.gpu_times.mean()]

    def check_correctness(self, algo_number) -> bool:
        try:
            cu = cutensor.contraction(self.alpha, self.a, self.mode_a, self.b, self.mode_b, self.beta, self.c, self.mode_c, algo = algo_number)
            to = torch.tensordot(self.atorch, self.btorch, dims = self.dimensions.tdotConDim)
        except RuntimeError as e:
            print(str(e) + " - Correctness check Err (CUDA ERROR: SEGMENT not initialized usually due to OOM)")
            return False
        except:
            print("Error in Correctness")
            return False
        
        # cuq = contract(self.cqinp, self.atorch, self.btorch)

        # if numpy.array_equal(cupy.asnumpy(cu), to.numpy) and numpy.array_equal(to.numpy, cuq.numpy) and numpy.array_equal(cuq.numpy, to.numpy):
        if numpy.allclose(cupy.asnumpy(cu),to.cpu().numpy(), atol=1e-3, rtol=1e-3):
            return True
        else:
            return False
    
    def parse_contype_einsum(self, con_type) -> list:
        modified_con_type = con_type.replace("*", ",")
        return modified_con_type
    
    def profile_all(self) -> list[str, list, list, list, list, list, list, list, list, bool, float, float, list]:
        if self.hasCrashed:
            return self.generate_memory_allocation_failure_return()
        cutensor_default = self.profile_cutensor(-1)
        cutensor_ttgt = self.profile_cutensor(-2)
        cutensor_tgett = self.profile_cutensor(-3)
        cutensor_gett = self.profile_cutensor(-4)
        cutensor_default_patient = self.profile_cutensor(-6)
        # cuquantum = self.profile_cuquantum()
        cuquantum = [float('inf'), float('inf')]
        tensordot = self.profile_tensordot()
        einsum = self.profile_einsum(self.dimensions.con_type)

        correctness = self.check_correctness(-4)

        lowest_CPU = self.fastest_time([cutensor_default[0], cutensor_ttgt[0], cutensor_tgett[0], cutensor_gett[0], cutensor_default_patient[0], cuquantum[0], tensordot[0], einsum[0]])
        lowest_GPU = self.fastest_time([cutensor_default[1], cutensor_ttgt[1], cutensor_tgett[1], cutensor_gett[1], cutensor_default_patient[1], cuquantum[1], tensordot[1], einsum[1]])
        fastest_CPU_value = self.fastest_time_value([cutensor_default[0], cutensor_ttgt[0], cutensor_tgett[0], cutensor_gett[0], cutensor_default_patient[0], cuquantum[0], tensordot[0], einsum[0]])
        fastest_GPU_value = self.fastest_time_value([cutensor_default[1], cutensor_ttgt[1], cutensor_tgett[1], cutensor_gett[1], cutensor_default_patient[1], cuquantum[1], tensordot[1], einsum[1]])

        speedup_over_tdot = self.speedup(tensordot[0], tensordot[1], fastest_CPU_value, fastest_GPU_value)

        self.cleanup()

        return [self.contractionLabel, cutensor_default, cutensor_ttgt, cutensor_tgett, cutensor_gett, cutensor_default_patient, cuquantum, tensordot, einsum, correctness, lowest_CPU, lowest_GPU, speedup_over_tdot]

    def fastest_time(self, inp) -> int:
        return algorithms[inp.index(min(inp))]
    
    def fastest_time_value(self, inp) -> int:
        return min(inp)

    def speedup(self, tdot_CPU, tdot_GPU, fastest_CPU, fastest_GPU) -> list:
        return [tdot_CPU/fastest_CPU, tdot_GPU/fastest_GPU]
    
    def cleanup(self) -> None:
        del self.a
        del self.b
        del self.c
        del self.atorch
        del self.btorch
        del self.mode_a
        del self.mode_b
        del self.mode_c
        del self.alpha
        del self.beta
        del self.dtype
        del self.torchdType
        del self.cqinp
        del self.dimensions
        del self.extent

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
    
    def generate_memory_allocation_failure_return(self):
        return [self.contractionLabel, [float('inf'), float('inf')], [float('inf'), float('inf')], [float('inf'), float('inf')], [float('inf'), float('inf')], [float('inf'), float('inf')], [float('inf'), float('inf')], [float('inf'), float('inf')], [float('inf'), float('inf')], False, "None", "None", [0,0]]