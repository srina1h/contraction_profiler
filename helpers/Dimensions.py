class Dimensions:
    def __init__(self, adim: tuple, bdim: tuple, cdim: tuple, tdotConDim: tuple, con_type: str, dataType: str):
        if adim not None and bdim not None and cdim not None and tdotConDim not None and con_type not None and dataType not None:
            self.adim = adim
            self.bdim = bdim
            self.cdim = cdim
            self.tdotConDim = tdotConDim
            self.con_type = con_type
            self.dataType = dataType