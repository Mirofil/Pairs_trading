#%%
import os
import numpy as np 
import pandas as pd 
import datetime
import matplotlib.pyplot as plt
import timeit
import datetime
import scipy
import statsmodels
import seaborn as sns
import multiprocess as mp
from dateutil.relativedelta import relativedelta
from distancemethod import *
from helpers import *
from cointmethod import *
from config import *
from simulation import *
from simulations_database import *
pd.options.mode.chained_assignment = None
formation = (datetime.date(*[2018,1,1]), datetime.date(*[2018,1,7]))
trading = (formation[1], formation[1]+relativedelta(days=3))

#%%#
#load all the time series retrieved
# files = os.listdir(data_path)
# #we exclude CLOAKBTC because theres some data-level mixed types mistake that breaks prefilter and it would get deleted anyways
# #it also breakts at ETHBTC (I manually deleted the first wrong part in Excel)
# paths = [data_path + x for x in files if x not in ['BTCUSDT.csv', 'ETHUSDT.csv', 'CLOAKBTC.csv']]
# names = [file.partition('.')[0] for file in files]
# df = pd.read_csv(paths[0])
# idx=pd.IndexSlice
#%%#
#EXTENSION needs cutoff 0.0 because we only download those that gets through cutoff 0.7 anyways..
# x=prefilter(paths, cutoff=0.7)
# np.save(version+'prefiltered0_7', x)
x=np.load('NEWprefiltered0_7.npy')

#%%#
# y=preprocess(x[:,0], first_n=0, freq='5T')
# y.to_pickle(version+'preprocessed5T0_7.pkl')
y=pd.read_pickle('NEWpreprocessed5T0_7.pkl')

#%%
# #13s per iteration (local)
simulate(scenario1)
#16s per iteration (local)
simulate(scenario1_coint)
#1min40s per iteration (sometimes down to 40)
simulate(scenario3)
#1min 10s per iteration (local) - extra volatile
simulate(scenario3_coint)

simulate(scenario1_nolag)
simulate(scenario3_nolag)
#NO TX SCENARIOS
simulate(scenario2)
simulate(scenario2_coint)
simulate(scenario4)
simulate(scenario4_coint)

# #MINUTE
simulate(scenario5, num_of_processes=35)
simulate(scenario6, num_of_processes=35)
simulate(scenario5_nolag, num_of_processes=35)

simulate(scenario5_coint, num_of_processes=35)
simulate(scenario6_coint, num_of_processes=35)
simulate(scenario5_coint_nolag, num_of_processes=35)

#5 MINUTE
simulate(scenario7, num_of_processes=3)
simulate(scenario8, num_of_processes=3)
simulate(scenario7_nolag, num_of_processes=3)

simulate(scenario7_coint, num_of_processes=35)
simulate(scenario8_coint, num_of_processes=35)
simulate(scenario7_coint_nolag, num_of_processes=35)


#STOPLOSS
stoploss()
#%%
#COINTEGRATION TESTING
start=datetime.datetime.now()
coint_head = pick_range(y, formation[0], formation[1])
# find_integrated(coint_head, num_of_processes=1)
k=cointegration(find_integrated(coint_head, num_of_processes=1), num_of_processes=3)
end=datetime.datetime.now()
print('Cointegrations were found in: ' + str(end-start))
#%%
short_y=pick_range(y, formation[0], trading[1])
start=datetime.datetime.now()
coint_spreads = coint_spread(short_y, [item[0] for item in k], timeframe=formation, betas = [item[1] for item in k])
coint_spreads.sort_index(inplace=True)
end=datetime.datetime.now()
print('Cointegrations spreads were done in: ' + str(end-start))
#%%
start=datetime.datetime.now()
num_of_processes=3
#pool = mp.Pool(num_of_processes)
split = np.array_split(coint_spreads, num_of_processes)
split = [pd.DataFrame(x) for x in split]
args_dict = {'trading':trading, 'formation':formation, 'threshold':2, 'lag':1, 'stoploss':100, 'num_of_processes':num_of_processes}
args = [args_dict['trading'], args_dict['formation'], args_dict['threshold'], args_dict['lag'], args_dict['stoploss'], args_dict['num_of_processes']]
full_args = [[split[i], *args] for i in range(len(split))]
# results = pool.starmap(signals, full_args)
# results=pd.concat(results)
# pool.close()
# pool.join()
coint_signal = signals(coint_spreads, timeframe = args_dict['trading'], formation = args_dict['formation'],lag = args_dict['lag'], stoploss = args_dict['stoploss'], num_of_processes=args_dict['num_of_processes'])
#results=signals(coint_spreads, timeframe = args_dict['trading'], formation = args_dict['formation'],lag = args_dict['lag'], stoploss = args_dict['stoploss'], num_of_processes=1)
end=datetime.datetime.now()
print('Signals were done in: ' + str(end-start))
# new = (results == coint_signal)
# new[new['Signals']==False]
#%%
start=datetime.datetime.now()
coint_signal = signals_numeric(coint_signal)
weights_from_signals(coint_signal, cost=0.003)
end=datetime.datetime.now()
print('Weight from signals was done in: ' + str(end-start))
#%%
#look at LTCxNEO on 12/29 for confirmation
start=datetime.datetime.now()
propagate_weights(coint_signal, formation)
end=datetime.datetime.now()
print('Weight propagation was done in: ' + str(end-start))

#%%
start=datetime.datetime.now()
calculate_profit(coint_signal, cost=0.003)
end=datetime.datetime.now()
print('Profit calculation was done in: ' + str(end-start))
#%%
#DISTANCE TESTING
#we take timeframe corresponding to Formation period when finding the lowest SSDs
start=datetime.datetime.now()
head = pick_range(y, formation[0], formation[1])
distances = distance(head, num = 20)
end=datetime.datetime.now()
print('Distances were found in: ' + str(end-start))
#%%
start=datetime.datetime.now()
short_y = pick_range(y, formation[0], trading[1])
spreads=distance_spread(short_y,distances[2], formation)
end=datetime.datetime.now()
print('Distance spreads were found in: ' + str(end-start))
# this is some technical detail needed later?
spreads.sort_index(inplace=True) 
#%%
start=datetime.datetime.now()
dist_signal=signals(spreads, timeframe=trading,formation=formation, lag = 1, num_of_processes=3)
weights_from_signals(dist_signal, cost=0.003)
end=datetime.datetime.now()
print('Distance signals were found in: ' + str(end-start))
#%%
start=datetime.datetime.now()
propagate_weights(dist_signal, formation)
end=datetime.datetime.now()
print('Weight propagation was done in: ' + str(end-start))
#%%
start=datetime.datetime.now()
calculate_profit(dist_signal, cost=0.003)
end=datetime.datetime.now()
print('Profit calculation was done in: ' + str(end-start))
#%%
dist_signal.to_pickle('test_dist.pkl')

