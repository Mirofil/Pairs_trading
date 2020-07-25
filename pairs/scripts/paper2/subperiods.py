import pandas as pd

class Subperiod:
    def __init__(self, start_date:str, end_date:str, name:str, preferred_txcost=0.003, table_name="Unknown"):
        super().__init__()
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.name = name
        self.preferred_txcost = preferred_txcost
        self.table_name = table_name

nineties = Subperiod(start_date='1990/1/1', end_date='2000/3/1', name='nineties', preferred_txcost=0.0035, table_name = '1990-2000')
dotcom = Subperiod(start_date='2000/3/1', end_date='2002/10/1', name='dotcom', preferred_txcost=0.003, table_name = '2000-2002')
inbetween_crises = Subperiod(start_date='2002/10/1', end_date='2007/08/01', name ='inbetween_crises', preferred_txcost=0.003, table_name='2002-2007')
financial_crisis = Subperiod(start_date = '2007/08/1', end_date = '2009/06/01', name='financial_crisis', preferred_txcost=0.003, table_name = '2007-2009')
modern = Subperiod(start_date='2009/06/01', end_date='2020/02/20', name='modern', preferred_txcost=0.0026, table_name = '2009-2020')
all_history = Subperiod(start_date='1990/1/1', end_date='2020/1/1', name='all_history', preferred_txcost=0.003, table_name = '1990-2020')

covid = Subperiod(start_date='2020/2/20', end_date='2020/6/1', name='covid', preferred_txcost = 0.0026, table_name='2020 (Covid)')