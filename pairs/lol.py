import numpy as np 
import pandas as pd 
import os
import datetime
import matplotlib.pyplot as plt
from config import *
from collections import namedtuple
from helpers import pick_range

#%%

def sliced_norm(df, pair, column, timeframe):
    sliced = pick_range(df, timeframe[0], timeframe[1])
    diff = df.loc[pair[0], column]-df.loc[pair[1], column]
    mean = (sliced.loc[pair[0], column]-sliced.loc[pair[1], column]).mean()
    std = (sliced.loc[pair[0], column]-sliced.loc[pair[1], column]).std()
    return ((diff-mean)/std).values

def distance_spread(df, viable_pairs, timeframe):
    """ The spread is calculated as the difference between the cumulatively summed 
    price series of a pair """
    pairs = df.index.unique(0)
    spreads = []
    for pair in viable_pairs:
        #labels will be IOTAADA rather that IOTABTCADABTC, 
        #so we remove the last three characters
        first = pair[0][:-3]
        second = pair[1][:-3]
        composed = first+'x'+second
        multiindex = pd.MultiIndex.from_product([[composed], df.loc[pair[0]].index],names = ['Pair', 'Time'])
        newdf = pd.DataFrame(index = multiindex)
        newdf['1Weights']=0
        newdf['2Weights']=0
        newdf['Profit'] = 0
        newdf['normLogReturns']= sliced_norm (df, pair, 'logReturns', timeframe)
        newdf['1Price']=df.loc[pair[0], 'Price'].values
        newdf['2Price']=df.loc[pair[1], 'Price'].values
        newdf['Spread'] = (df.loc[pair[0], 'Price']-df.loc[pair[1], 'Price']).values
        newdf['normSpread']=((newdf['Spread']-pick_range(newdf, *timeframe)['Spread'].mean())/pick_range(newdf, *timeframe)['Spread'].std()).values
        first = df.loc[pair[0]]
        first.columns = ["1"+x for x in first.columns]
        second = df.loc[pair[0]]
        second.columns = ["2"+x for x in second.columns]
        reindexed = (pd.concat([first,second], axis=1)).set_index(multiindex)
        
        #normPriceOld = reindexed.normPrice
        #reindexed.loc[:,'normPrice'] = (reindexed.loc[:,'normPrice']-reindexed.loc[:,'normPrice'].mean())/reindexed.loc[:,'normPrice'].std()
        #possible deletion of useless columns to save memory.. 
        # but maybe should be done earlier? Only normPrice 
        # should be relevant since its the spread at this point
        # reindexed.drop(['Volume', 'Close', 'Returns'], axis = 1)
        #reindexed['normPriceOld'] = normPriceOld
        spreads.append(newdf)
    return pd.concat(spreads)


#%%
