import datetime

startdate = datetime.date(*[2018,1,1])
enddate = datetime.date(*[2019,9,1])
volume_cutoff = 0.7
version='NEW'
desc=['Mean', 'Total profit','Std', 'Sharpe','Sortino', 'VaR', 'Calmar',
     'Number of trades', 'Roundtrip trades', 'Avg length of position', 'Pct of winning trades', 'Max drawdown', 'Cumulative profit']
data_path = version+'concatenated_price_data\ '[:-1]
save = version+'results\ '[:-1]
save_path_graphs = version+'graphs\ '[:-1]
save_path_tables = version+'tables\ '[:-1]
NUMOFPROCESSES=35
