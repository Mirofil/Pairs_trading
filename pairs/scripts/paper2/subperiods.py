import pandas as pd

class Subperiod:
    def __init__(self, start_date:str, end_date:str, name:str):
        super().__init__()
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.name = name

nineties = Subperiod(start_date='1990/1/1', end_date='2000/3/1', name='nineties')
dotcom = Subperiod(start_date='2000/3/1', end_date='2002/10/1', name='dotcom')
inbetween_crises = Subperiod(start_date='2002/10/1', end_date='2007/08/01', name ='inbetween_crises')
financial_crisis = Subperiod(start_date = '2007/08/1', end_date = '2009/06/01', name='financial_crisis')
modern = Subperiod(start_date='2009/06/01', end_date='2020/01/01', name='modern')