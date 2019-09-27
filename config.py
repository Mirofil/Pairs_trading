import datetime

startdate = datetime.date(*[2018,1,1])
enddate = datetime.date(*[2019,1,1])
volume_cutoff = 0.7
data_folder = "C:\Bach\concatenated_price_data"
desc=['Mean', 'Total profit','Std', 'Sharpe','Sortino', 'VaR', 'Calmar',
     'Number of trades', 'Roundtrip trades', 'Avg length of position', 'Pct of winning trades', 'Max drawdown', 'Cumulative profit']