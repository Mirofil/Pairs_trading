#%%
import numpy as np 
import pandas as pd 
import os
import datetime
import matplotlib.pyplot as plt
import timeit
import datetime
import scipy
import statsmodels
import seaborn as sns
from dateutil.relativedelta import relativedelta
from distancemethod import *
from helpers import *
from cointmethod import *
from config import *
from simulation import *
from simulations_database import *
formation = (datetime.date(*[2018,1,1]), datetime.date(*[2018,5,1]))
trading = (formation[1], formation[1]+relativedelta(months=2))

#%%#
#load all the time series retrieved
admissible=pairs=['MDABTC', 'MTLBTC', 'LINKBTC', 'DASHBTC', 'OMGBTC', 'ENJBTC', 'XMRBTC',
       'ARNBTC', 'ZRXBTC', 'QTUMBTC', 'WAVESBTC', 'WTCBTC', 'ETCBTC',
       'IOTABTC', 'XVGBTC', 'LTCBTC', 'ADABTC', 'NEOBTC', 'BNBBTC', 'EOSBTC',
       'XRPBTC', 'TRXBTC', 'ETHBTC']
files = os.listdir(data_folder)
#we exclude CLOAKBTC because theres some data-level mixed types mistake that breaks prefilter and it would get deleted anyways
#it also breakts at ETHBTC (I manually deleted the first wrong part in Excel)
paths = ['C:\Bach\concatenated_price_data\ '[:-1] + x for x in files if x not in ['BTCUSDT.csv', 'ETHUSDT.csv', 'CLOAKBTC.csv']]
names = [file.partition('.')[0] for file in files]
df = pd.read_csv(paths[0])
idx=pd.IndexSlice
#%%#
#rerunning is computationally intensive
x=prefilter(paths, cutoff=0.0)
#np.save('prefiltered0_0', x)
x=np.load('prefiltered0_7.npy')

#%%#
#y=preprocess(x[:,0], first_n=0, freq='1H')
#y.to_pickle('preprocessedH0_7.pkl')
y=pd.read_pickle('preprocessedD0_7.pkl')
#%%#
#simulate(scenario3_coint)
# simulate(scenario2_coint)
# simulate(scenario4)
# simulate(scenario4_coint)

#%%
#COINTEGRATION TESTING
coint_head = pick_range(y, formation[0], formation[1])
k=cointegration(find_integrated(coint_head))

#%%
coint_spreads = coint_spread(y, [item[0] for item in k], timeframe=formation, betas = [item[1] for item in k])
coint_spreads.sort_index(inplace=True)

#%%
coint_signal = signals(coint_spreads, timeframe = trading, formation = formation,lag = 1, stoploss = 100)

#%%
coint_signal = signals_numeric(coint_signal)
weights_from_signals(coint_signal, cost=0.003)
#%%
#look at LTCxNEO on 12/29 for confirmation
propagate_weights(coint_signal, formation)

#%%
calculate_profit(coint_signal, cost=0.003)
#%%

#%%

#%%
#DISTANCE TESTING
#we take timeframe corresponding to Formation period when finding the lowest SSDs
head = pick_range(y, formation[0], formation[1])
distances = distance(head, num = 20)
spreads=distance_spread(y,distances[2], formation)
# this is some technical detail needed later?
spreads.sort_index(inplace=True) 
#%%
dist_signal=signals(spreads, timeframe=trading,formation=formation, lag = 1)
weights_from_signals(dist_signal, cost=0.003)

#%%
propagate_weights(dist_signal, formation)
#%%
calculate_profit(dist_signal, cost=0.003)
#%%
dist_signal.to_pickle('test_dist.pkl')

