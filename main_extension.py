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
#EXTENSION needs cutoff 0.0 because we only download those that gets through cutoff 0.7 anyways..
x=prefilter(paths, cutoff=0.7)
#np.save('prefiltered0_0', x)
x=np.load('prefiltered0_7.npy')

#%%#
#y=preprocess(x[:,0], first_n=0, freq='1H')
#y.to_pickle('preprocessedH0_7.pkl')
y=pd.read_pickle('preprocessedD0_7.pkl')

#%%
simulate(scenario1)
simulate(scenario1_coint)
simulate(scenario3)
simulate(scenario3_coint)

#Order sensitive!
starting_date=datetime.date(*[2018,1,1])
ending_date = datetime.date(*[2019,1,1])
#DAILY
scenario1={"freq":"1D",'lag':1, 'txcost':0.003, 'training_delta':[2,0,0], 'cutoff':0.7, 'formation_delta':[4,0,0], 'start':starting_date, 'end':ending_date, 'jump':[1,0,0], 'methods':['dist'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':"scenario1"}
scenario2={"freq":"1D",'lag':1, 'txcost':0.000, 'training_delta':[2,0,0], 'cutoff':0.7, 'formation_delta':[4,0,0], 'start':starting_date, 'end':ending_date, 'jump':[1,0,0], 'methods':['dist'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':'scenario2'}
#changed cutoff
scenario1_1={"freq":"1D",'lag':1, 'txcost':0.003, 'training_delta':[2,0,0], 'cutoff':0.0, 'formation_delta':[4,0,0], 'start':starting_date, 'end':ending_date, 'jump':[1,0,0], 'methods':['dist'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':"scenario1_1"}
scenario2_1={"freq":"1D",'lag':1, 'txcost':0.000, 'training_delta':[2,0,0], 'cutoff':0.0, 'formation_delta':[4,0,0], 'start':starting_date, 'end':ending_date, 'jump':[1,0,0], 'methods':['dist'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':'scenario2_2'}
#COINT version
scenario1_coint={"freq":"1D",'lag':1, 'txcost':0.003, 'training_delta':[2,0,0], 'cutoff':0.7, 'formation_delta':[4,0,0], 'start':starting_date, 'end':ending_date, 'jump':[1,0,0], 'methods':['coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':"scenario1"}
scenario2_coint={"freq":"1D",'lag':1, 'txcost':0.000, 'training_delta':[2,0,0], 'cutoff':0.7, 'formation_delta':[4,0,0], 'start':starting_date, 'end':ending_date, 'jump':[1,0,0], 'methods':['coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':'scenario2'}
scenario1_1={"freq":"1D",'lag':1, 'txcost':0.003, 'training_delta':[2,0,0], 'cutoff':0.0, 'formation_delta':[4,0,0], 'start':starting_date, 'end':ending_date, 'jump':[1,0,0], 'methods':['coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':"scenario1_1"}
scenario2_1={"freq":"1D",'lag':1, 'txcost':0.000, 'training_delta':[2,0,0], 'cutoff':0.0, 'formation_delta':[4,0,0], 'start':starting_date, 'end':ending_date, 'jump':[1,0,0], 'methods':['coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':'scenario2_2'}

#NOLAG
scenario1_nolag={"freq":"1D",'lag':0, 'txcost':0.003, 'training_delta':[2,0,0], 'cutoff':0.7, 'formation_delta':[4,0,0], 'start':starting_date, 'end':ending_date, 'jump':[1,0,0], 'methods':['dist', 'coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':"scenario1_nolag"}
scenario2_nolag={"freq":"1D",'lag':0, 'txcost':0.000, 'training_delta':[2,0,0], 'cutoff':0.7, 'formation_delta':[4,0,0], 'start':starting_date, 'end':ending_date, 'jump':[1,0,0], 'methods':['dist', 'coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':'scenario2_nolag'}

#HOURLY
scenario3={"freq":"1H",'lag':1, 'txcost':0.003, 'training_delta':[0,10,0], 'cutoff':0.7, 'formation_delta':[0,20,0], 'start':starting_date, 'end':ending_date, 'jump':[0,10,0], 'methods':['dist'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':'scenario3'}
scenario4={"freq":"1H",'lag':1, 'txcost':0.000, 'training_delta':[0,10,0], 'cutoff':0.7, 'formation_delta':[0,20,0], 'start':starting_date, 'end':ending_date, 'jump':[0,10,0], 'methods':['dist'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':'scenario4'}
#changed cutoff
scenario3_1={"freq":"1H",'lag':1, 'txcost':0.003, 'training_delta':[0,10,0], 'cutoff':0.0, 'formation_delta':[0,20,0], 'start':starting_date, 'end':ending_date, 'jump':[0,10,0], 'methods':['dist'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':'scenario3_1'}
scenario4_1={"freq":"1H",'lag':1, 'txcost':0.000, 'training_delta':[0,10,0], 'cutoff':0.0, 'formation_delta':[0,20,0], 'start':starting_date, 'end':ending_date, 'jump':[0,10,0], 'methods':['dist'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':'scenario4_1'}

#NOLAG
scenario3_nolag={"freq":"1H",'lag':0, 'txcost':0.003, 'training_delta':[0,10,0], 'cutoff':0.7, 'formation_delta':[0,20,0], 'start':starting_date, 'end':ending_date, 'jump':[0,10,0], 'methods':['dist', 'coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':'scenario3_nolag'}
scenario4_nolag={"freq":"1H",'lag':0, 'txcost':0.000, 'training_delta':[0,10,0], 'cutoff':0.7, 'formation_delta':[0,20,0], 'start':starting_date, 'end':ending_date, 'jump':[0,10,0], 'methods':['dist', 'coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':'scenario4_nolag'}
#COINT version
scenario3_coint={"freq":"1H",'lag':1, 'txcost':0.003, 'training_delta':[0,10,0], 'cutoff':0.7, 'formation_delta':[0,20,0], 'start':starting_date, 'end':ending_date, 'jump':[0,10,0], 'methods':['coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':'scenario3'}
scenario4_coint={"freq":"1H",'lag':1, 'txcost':0.000, 'training_delta':[0,10,0], 'cutoff':0.7, 'formation_delta':[0,20,0], 'start':starting_date, 'end':ending_date, 'jump':[0,10,0], 'methods':['coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':'scenario4'}
scenario3_1_coint={"freq":"1H",'lag':1, 'txcost':0.003, 'training_delta':[0,15,0], 'cutoff':0.0, 'formation_delta':[1,0,0], 'start':starting_date, 'end':ending_date, 'jump':[1,0,0], 'methods':['coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':'scenario3_1'}
scenario4_1_coint={"freq":"1H",'lag':1, 'txcost':0.000, 'training_delta':[0,15,0], 'cutoff':0.0, 'formation_delta':[1,0,0], 'start':starting_date, 'end':ending_date, 'jump':[1,0,0], 'methods':['coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':'scenario4_1'}

#VARIOUS DELTA SCHEMES
scenario3_nolag1={"freq":"1H",'lag':0, 'txcost':0.003, 'training_delta':[0,15,0], 'cutoff':0.7, 'formation_delta':[1,0,0], 'start':starting_date, 'end':ending_date, 'jump':[0,15,0], 'methods':['dist', 'coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':'scenario3_nolag1'}
scenario1_nolag1={"freq":"1D",'lag':0, 'txcost':0.003, 'training_delta':[3,0,0], 'cutoff':0.7, 'formation_delta':[6,0,0], 'start':starting_date, 'end':ending_date, 'jump':[1,0,0], 'methods':['dist', 'coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':"scenario1_nolag1"}
scenario11={"freq":"1D",'lag':1, 'txcost':0.003, 'training_delta':[3,0,0], 'cutoff':0.7, 'formation_delta':[6,0,0], 'start':starting_date, 'end':ending_date, 'jump':[1,0,0], 'methods':['dist', 'coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':"scenario11"}
scenario31={"freq":"1H",'lag':1, 'txcost':0.003, 'training_delta':[0,15,0], 'cutoff':0.7, 'formation_delta':[1,0,0], 'start':starting_date, 'end':ending_date, 'jump':[0,15,0], 'methods':['dist', 'coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':'scenario31'}


#%%
#COINTEGRATION TESTING
start=datetime.datetime.now()
coint_head = pick_range(y, formation[0], formation[1])
k=cointegration(find_integrated(coint_head, num_of_processes=1), num_of_processes=3)
end=datetime.datetime.now()
print('Cointegrations were found in: ' + str(end-start))
#%%
start=datetime.datetime.now()
coint_spreads = coint_spread(y, [item[0] for item in k], timeframe=formation, betas = [item[1] for item in k])
coint_spreads.sort_index(inplace=True)
end=datetime.datetime.now()
print('Cointegrations spreads were done in: ' + str(end-start))
#%%
start=datetime.datetime.now()
coint_signal = signals(coint_spreads, timeframe = trading, formation = formation,lag = 1, stoploss = 100, num_of_processes=3)
end=datetime.datetime.now()
print('Signals were done in: ' + str(end-start))
#%%
coint_signal = signals_numeric(coint_signal)
weights_from_signals(coint_signal, cost=0.003)
#%%
#look at LTCxNEO on 12/29 for confirmation
propagate_weights(coint_signal, formation)

#%%
calculate_profit(coint_signal, cost=0.003)


#%%
new=[]
for split in np.array_split(list(coint_spreads.loc[idx[:, trading[0]:trading[1]], :].groupby(level=0)),3):
       for x in split:
              new.append(x[1])
new=pd.concat(new)
new=new.loc[:, 'normSpread']
for name, df in new.groupby(level=0):
       ind=df.index
       for i in range(5):
              df.loc[ind[i], 'normSpread']
#%%
new.loc[0]

#%%
#DISTANCE TESTING
#we take timeframe corresponding to Formation period when finding the lowest SSDs
start=datetime.datetime.now()
head = pick_range(y, formation[0], formation[1])
distances = distance(head, num = 20)
end=datetime.datetime.now()
print('Distances were found in: ' + str(end-start))
start=datetime.datetime.now()
spreads=distance_spread(y,distances[2], formation)
end=datetime.datetime.now()
print('Distance spreads were found in: ' + str(end-start))
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

