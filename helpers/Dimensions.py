class Dimensions:
    def __init__(self, adim: tuple, bdim: tuple, cdim: tuple, tdotConDim: tuple(list, list), con_type: str, dataType: str):
        self.adim = adim
        self.bdim = bdim
        self.cdim = cdim
        self.tdotConDim = tdotConDim
        self.con_type = con_type
        self.dataType = dataType