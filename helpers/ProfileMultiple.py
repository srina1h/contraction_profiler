from helpers.ContractionProfiler import *
from helpers.Dimensions import *
import pandas as pd
import ast

class ProfileMultiple:
    def __init__(self, file_input: str, output_filepath: str, type: str, baseline: str) -> None:
        self.file_input = file_input
        if type == "csv":
            self.csv_file = file_input
            self.data = pd.read_csv(self.csv_file)
        elif type == "xlsx":
            self.xlsx_file = file_input
            self.data = pd.read_excel(self.xlsx_file)
        self.output_filepath = output_filepath
        self.results = []
        self.baseline = baseline
        self.profile()
        self.export()

    def profile(self) -> None:
        for index, row in self.data.iterrows():
            self.dimensions = Dimensions(ast.literal_eval(row['adim']), ast.literal_eval(row['bdim']), ast.literal_eval(row['cdim']), ast.literal_eval(row['condim']), row['type'], row['dtype'])
            profiler = ContractionProfiler(self.dimensions, row['label'], self.baseline)
            result = profiler.profile_all()
            self.results.append(result)
            del profiler
            del result
        
    def export(self) -> None:
        print(len(self.results))
        df = pd.DataFrame(columns = ['adim', 'bdim', 'cdim', 'condim', 'con_label','default_CPU', 'default_GPU', 'ttgt_CPU', 'ttgt_GPU', 'tgett_CPU', 'tgett_GPU', 'gett_CPU', 'gett_GPU', 'default_patient_CPU', 'default_patient_GPU', 'cuquantum_CPU', 'cuquantum_GPU', 'tensordot_CPU', 'tensordot_GPU', 'einsum_CPU', 'einsum_GPU', 'cutensor_direct_CPU', 'cutensor_direct_GPU', 'correctness', 'fastest_CPU', 'fastest_GPU', 'speedup_CPU', 'speedup_GPU', 'theory_mem', 'torch_mem'])
        for result in self.results:
            # print(len(result))
            df = df._append({'adim': self.dimensions.adim, 'bdim': self.dimensions.bdim, 'cdim': self.dimensions.cdim, 'condim': self.dimensions.tdotConDim, 'con_label': result[0], 'default_CPU': result[1][0], 'default_GPU': result[1][1], 'ttgt_CPU': result[2][0], 'ttgt_GPU': result[2][1], 'tgett_CPU': result[3][0], 'tgett_GPU': result[3][1], 'gett_CPU': result[4][0], 'gett_GPU': result[4][1], 'default_patient_CPU': result[5][0], 'default_patient_GPU': result[5][1], 'cuquantum_CPU': result[6][0], 'cuquantum_CPU': result[6][1], 'tensordot_CPU': result[7][0], 'tensordot_GPU': result[7][1], 'einsum_CPU': result[8][0], 'einsum_GPU': result[8][1], 'cutensor_direct_CPU': result[9][0], 'cutensor_direct_GPU': result[9][1], 'correctness': result[10], 'fastest_CPU': result[11], 'fastest_GPU': result[12], 'speedup_CPU': result[13][0], 'speedup_GPU': result[13][1], 'theory_mem': result[14], 'torch_mem': result[15]}, ignore_index = True)
        df.to_excel(self.output_filepath)