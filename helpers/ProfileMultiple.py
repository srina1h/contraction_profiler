from helpers import ContractionProfiler
from helpers import Dimensions
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
            dimensions = Dimensions(ast.literal_eval(row['adim']), ast.literal_eval(row['bdim']), ast.literal_eval(row['cdim']), ast.literal_eval(row['condim']), ast.literal_eval(row['type']), ast.literal_eval(row['dataType']))
            profiler = ContractionProfiler(dimensions, row['label'])
            result = profiler.profile()
            self.results.append(result)
        
    def export(self) -> None:
        self.results.to_csv(self.output_filepath, index = False)
    
    def get_results(self) -> pd.DataFrame:
        return self.results