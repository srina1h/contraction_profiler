from helpers.ContractionProfiler import *
from helpers.Dimensions import *
import pandas as pd
import ast

class ProfileMultiple:
    def __init__(self, file_input: str, output_filepath: str, type: str) -> None:
        self.file_input = file_input
        if type == "csv":
            self.csv_file = file_input
            self.data = pd.read_csv(self.csv_file)
        elif type == "xlsx":
            self.xlsx_file = file_input
            self.data = pd.read_excel(self.xlsx_file)
        self.output_filepath = output_filepath
        self.results = []
        self.profile()
        self.export()

    def profile(self) -> None:
        for index, row in self.data.iterrows():
            dimensions = Dimensions(ast.literal_eval(row['adim']), ast.literal_eval(row['bdim']), ast.literal_eval(row['cdim']), ast.literal_eval(row['condim']), row['type'], row['dtype'])
            profiler = ContractionProfiler(dimensions, row['label'])
            result = profiler.profile_all()
            self.results.append(result)
        
    def export(self) -> None:
        print(len(self.results))
        df = pd.DataFrame(columns = ['con_label','default_CPU', 'default_GPU', 'ttgt_CPU', 'ttgt_GPU', 'tgett_CPU', 'tgett_GPU', 'gett_CPU', 'gett_GPU', 'default_patient_CPU', 'default_patient_GPU', 'tensordot_CPU', 'tensordot_GPU', 'correctness', 'fastest_CPU', 'fastest_GPU'], index = [i for i in range(len(self.results))])
        for result in self.results:
            print(len(result))
            for i in result:
                df = df._append({'con_label': i[0], 'default_CPU': i[1][0], 'default_GPU': i[1][1], 'ttgt_CPU': i[2][0], 'ttgt_GPU': i[2][1], 'tgett_CPU': i[3][0], 'tgett_GPU': i[3][1], 'gett_CPU': i[4][0], 'gett_GPU': i[4][1], 'default_patient_CPU': i[5][0], 'default_patient_GPU': i[5][1], 'tensordot_CPU': i[6][0], 'tensordot_GPU': i[6][1], 'correctness': i[7], 'fastest_CPU': i[8], 'fastest_GPU': i[9]}, ignore_index = True)
        df.to_csv(self.output_filepath)