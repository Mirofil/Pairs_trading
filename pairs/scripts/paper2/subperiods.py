import pandas as pd

class Subperiod:
    def __init__(self, start_date:str, end_date:str):
        super().__init__()
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

nineties = Subperiod(start_date='1990/1/1', end_date='2000/1/1')
dotcom = Subperiod(start_date='2000/3/1', end_date='2002/10/1')