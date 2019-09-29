import numpy as np 
import pandas as pd 
import os
import datetime
import matplotlib.pyplot as plt
import scipy
import statsmodels
from dateutil.relativedelta import relativedelta
from config import *
import pickle
from contextlib import contextmanager
import os
import shutil
import multiprocess as mp
#%%
def pick_range(y, start, end):
    """ Slices preprocessed index-wise to achieve y[start:end], taking into account the MultiIndex"""
    past_start = (y.index.levels[1]>pd.to_datetime(start))
    before_end = (y.index.levels[1]<=pd.to_datetime(end))
    mask = (past_start) & (before_end)
    return y.groupby(level=0).apply(lambda x: x.loc[mask]).droplevel(level=0)

def name_from_path(path):
    ''' Goes from stuff like C:\Bach\concat_data\[pair].csv to [pair].csv '''
    name= path.split("\\")[-1].partition('.')[0]
    return name
def path_from_name(name):
    """ Goes from stuff like [pair].csv to C:\Bach\concat_data\[pair].csv"""
    path = data_path + name + '.csv'
    return name
def prefilter(paths, start=startdate,end=enddate, cutoff=0.7):
    """ Prefilters the time series so that we have only moderately old pairs (listed past startdate)
    and uses a volume percentile cutoff. The output is in array (pair, its volume) """
    idx=pd.IndexSlice
    admissible = []
    for i in range(len(paths)):
        print(paths[i])
        df = pd.read_csv(paths[i])
        df.rename({'Opened':'Date'}, axis='columns', inplace=True)
        # filters out pairs that got listed past startdate
        if (pd.to_datetime(df.iloc[0].Date)<pd.to_datetime(start)) and (pd.to_datetime(df.iloc[-1].Date) > pd.to_datetime(end)):
            #the Volume gets normalized to BTC before sorting
            df=df.set_index('Date')
            df=df.sort_index()
            admissible.append([paths[i], (df.loc[idx[str(start):str(end)]].Volume*df.loc[idx[str(start):str(end)]].Close).sum()])
    #sort by Volume and pick upper percentile
    admissible.sort(key = lambda x: x[1])
    admissible=admissible[int(np.round(len(admissible)*cutoff)):]
    return np.array(admissible)

def resample(df, freq='30T', start=startdate, fill=True):
    """ Our original data is 1-min resolution, so we resample it to arbitrary frequency.
    Close prices get last values, Volume gets summed. 
    Only indexes past startdate are returned to have a common start for all series 
    (since they got listed at various dates)"""
    df.index=pd.to_datetime(df.Date)
    # Close prices get resampled with last values, whereas Volume gets summed
    close = df.Close.resample(freq).last()
    df.Volume = df.Volume*df.Close
    newdf = df.resample(freq).agg({"Volume":np.sum})
    #log returns and normalization
    newdf['Close']=close
    if fill == True:
        newdf['Close'] = newdf['Close'].fillna(method="ffill")
    newdf['logClose'] = np.log(newdf['Close'])
    newdf['logReturns'] = (np.log(newdf['Close'])-np.log(newdf['Close'].shift(1))).values
    newdf['Price']=newdf['logReturns'].cumsum()
    return newdf[newdf.index > pd.to_datetime(start)]

def preprocess(paths,freq='60T', end = enddate, first_n = 15, start=startdate):
    """Finishes the preprocessing based on prefiltered paths. We filter out pairs that got delisted early
    (they need to go at least as far as enddate). Then all the eligible time series for pairs formation analysis
    are concated into one big DF with a multiIndex (pair, time)."""
    #paths are truncated for faster processing
    #pairs = [pd.read_csv(x) for x in paths]
    paths=paths[first_n:]
    preprocessed = []
    for i in range(len(paths)):
        df = pd.read_csv(paths[i])
        #The new Binance_fetcher API downloads Date as Opened instead..
        df.rename({'Opened':'Date'}, axis='columns', inplace=True)
        df = resample(df, freq, start=start)
        df = df.sort_index()
        #truncates the time series to a slightly earlier end date
        #because the last period is inhomogeneous due to pulling from API
        if df.index[-1] > pd.to_datetime(end):
            newdf=df[df.index<pd.to_datetime(end)]
            multiindex = pd.MultiIndex.from_product([[name_from_path(paths[i])], 
                list(newdf.index.values)], names=['Pair', 'Time'])
            preprocessed.append(newdf.set_index(multiindex))
    concat=pd.concat(preprocessed)
    #concat.groupby(level=0)['Price']=concat.groupby(level=0)['Price'].shift(0)-concat.groupby(level=0)['Price'][0]
    #this step has to be done here even though it thematically fits end of prefilter since its not fully truncated by date and we would have to at least subtract the first row but whatever
    #concat.groupby(level=0).apply(lambda x: x['Price']=x['logReturns'].cumsum())
    return pd.concat(preprocessed)  
 
def signals_numeric(olddf, copy=True):
    """ Prepares dummy variables so we can make a graph when the pair is open/close etc"""
    df=olddf.copy(deep=copy)
    for name,group in df.groupby(level=0):
        numeric = np.zeros((df.loc[name].shape[0]))
        numeric[(df.loc[name, 'Signals'] == 'Long') | (df.loc[name, 'Signals'] == 'keepLong')] = 1
        numeric[(df.loc[name, 'Signals'] == 'Short') | (df.loc[name, 'Signals'] == 'keepShort')] = -1
        df.loc[name, 'Numeric']=numeric
    return df

def signals_graph(df, pair, timeframe=None):
    if timeframe == None:
        df.loc[pair, 'Numeric'].plot()
    else:
        sliced= pick_range(df, timeframe[0], timeframe[1]).loc[pair]
        sliced['Numeric'].plot()

    return 1
def sliced_norm(df, pair, column, timeframe):
    """ normalizes a dataframe by timeframe slice (so the mean overall
     is not actually 0 etc) """
    sliced = pick_range(df, timeframe[0], timeframe[1])
    diff = df.loc[pair[0], column]-df.loc[pair[1], column]
    mean = (sliced.loc[pair[0], column]-sliced.loc[pair[1], column]).mean()
    std = (sliced.loc[pair[0], column]-sliced.loc[pair[1], column]).std()
    return ((diff-mean)/std).values

def weights_from_signals(df, cost = 0):
    """ Sets the initial weights on position open so they can be propagated"""
    df.loc[df['Signals']== 'Long', '1Weights']=-df.loc[df['Signals']== 'Long', 'SpreadBeta']*(1+cost)
    df.loc[df['Signals']== 'Long', '2Weights']=1*(1-cost)
    df.loc[df['Signals']== 'Short', '1Weights']=df.loc[df['Signals']== 'Short', 'SpreadBeta']*(1-cost)
    df.loc[df['Signals']== 'Short', '2Weights']=-1*(1+cost)
    df.loc[((df['Signals']=='sellLong') | (df['Signals']=='sellShort') | (df['Signals']=='Sell')), '1Weights']=0
    df.loc[((df['Signals']=='sellLong') | (df['Signals']=='sellShort') | (df['Signals']=='Sell')), '2Weights']=0

def propagate_weights(df, timeframe):
    """Propagates weights according to price changes
    Timeframe should be Formation """
    for name, group in df.groupby(level=0):
        end_of_formation = df.loc[name].index.get_loc(timeframe[1])
        temp_weights1 = group['1Weights'].to_list()
        temp_weights2 = group['2Weights'].to_list()
        return1 = (group['1Price']-group['1Price'].shift(1))
        return2 = (group['2Price']-group['2Price'].shift(1))
        #print(end_of_formation, len(group.index), name)
        for i in range(end_of_formation+1,len(group.index)):
            if group.iloc[i]['Signals'] in ['keepLong', 'keepShort']:
                #print(temp_weights1[i-1], temp_weights2[i-1])
                #print(group.index[i])
                #df.loc[(name,group.index[i]),'1Weights']=df.loc[(name, group.index[i-1]), '1Weights']*1.1
                #not sure if the indexes are matched correctly here
                temp_weights1[i]=temp_weights1[i-1]*(1+return1.iloc[i])
                temp_weights2[i]=temp_weights2[i-1]*(1+return2.iloc[i])
        df.loc[name, '1Weights']=temp_weights1
        df.loc[name, '2Weights']=temp_weights2

def propagate_weights2(df, timeframe):
    idx=pd.IndexSlice
    grouped=df.groupby(level=0)
    for name, group in df.groupby(level=0):
        end_of_formation = df.loc[name].index.get_loc(timeframe[1])
        return1 = (group['1Price']-group['1Price'].shift(1))
        return2 = (group['2Price']-group['2Price'].shift(1))
        mask = (df['Signals']=='keepLong')|(df['Signals']=='keepShort')
        #mask = (group['Signals']=='keepLong')|(group['Signals']=='keepShort')
        cumreturn1 = (return1+1).loc[idx[mask]].cumprod()
        cumreturn2 = (return2+1).cumprod()
        #print(len(mask))
        #print(df.loc[idx[name, mask], '1Weights'])
        #print(cumreturn1)
        #print(df.loc[idx[name, mask], '1Weights'])
        df.loc[idx[name, mask], '1Weights']= df.loc[idx[name, mask], '1Weights'].shift(1)*cumreturn1
        #df.loc[idx[name,mask],'1Weights']=5


def calculate_profit(df, cost = 0):
    """Inplace calculates the profit per period as well as a cumulative profit
    Be careful to have the same cost as weights_from_signals
    This function counts the cost in Profit, while w_f_s does it for weights
    So its not double counting or anything"""
    idx=pd.IndexSlice 
    mask = ((df.loc[idx[:, 'Signals']] == 'Long') | (df.loc[idx[:, 'Signals']] == 'Short'))
    #used for cumProfit propagation
    mask2 = ((df.loc[idx[:, 'Signals']] == 'Long') | (df.loc[idx[:, 'Signals']] == 'Short') | (df.loc[idx[:, 'Signals']] == 'keepShort') | (df.loc[idx[:, 'Signals']] == 'keepLong')|(df.loc[idx[:, 'Signals']] == 'Sell')|(df.loc[idx[:, 'Signals']] == 'sellShort')|(df.loc[idx[:, 'Signals']] == 'sellLong'))
    for name, group in df.groupby(level=0):
        returns1 = (group['1Price']-group['1Price'].shift(1).values)
        returns2 = (group['2Price']-group['2Price'].shift(1).values)
        temp = returns1+returns2
        df.loc[name, 'Profit']= df.loc[name, '1Weights'].shift(1)*returns1+df.loc[name, '2Weights'].shift(1)*returns2
        df.loc[idx[name, mask], 'Profit'] = -(df.loc[idx[name, mask], '1Weights'].abs()*cost+df.loc[idx[name, mask], '2Weights'].abs()*cost)
        df.loc[idx[name, mask2], 'cumProfit']=(df.loc[idx[name, mask2], 'Profit']).cumsum()+1

def latexsave(df, file, params=[]):
    with open(file+'.tex', 'w') as tf:
     tf.write(df.to_latex(*params, escape=False))

def drawdown(df):
    """Calculates the maximum drawdown. Window is just mean to be bigger than examined period"""
    window = 25000
    roll_max = df['cumProfit'].rolling(window, min_periods=1).max()
    daily_drawdown = df['cumProfit']/roll_max-1.0
    max_daily_drawdown = daily_drawdown.rolling(window, min_periods = 1).min()
    return max_daily_drawdown

def find_trades(df, timeframe=5):
    """ Identifies the periods where we actually trade the pairs"""
    idx=pd.IndexSlice
    starts = (df.loc[idx[:], 'Signals'] == 'Long')|(df.loc[idx[:], 'Signals'] == 'Short')
    ends = (df.loc[idx[:], 'Signals'] == 'sellLong')|(df.loc[idx[:], 'Signals'] == 'sellShort')
    if starts.sum()>ends.sum():
        ends = ends|(df.loc[idx[:], 'Signals'] == 'Sell')
    return (starts, ends)

def corrs(df):
    cols = ['Sharpe', 'Sortino', 'Calmar', 'VaR']
    arr = pd.DataFrame(columns = cols, index = cols)
    ps = pd.DataFrame(columns = cols, index = cols)
    df.replace([np.inf, -np.inf], np.nan)
    mask = (df['Sharpe'].notnull())&(df['Sortino'].notnull())&(df['VaR'].notnull())&(df['Calmar'].notnull())
    for i in range(len(cols)):
        for j in range(len(cols)):
            arr.loc[cols[i], cols[j]]=scipy.stats.spearmanr(df[cols[i]].loc[mask], df[cols[j]].loc[mask])[0]
            ps.loc[cols[i], cols[j]]=rhoci(scipy.stats.spearmanr(df[cols[i]].loc[mask], df[cols[j]].loc[mask])[0], len(df[cols[i]].loc[mask]))
    return (arr, ps)
def descriptive_stats(df, timeframe = 5, freq = 'daily', riskfree = 0.02, tradingdays = 60, nonzero=False, trades_nonzero = False):
    """Input: one period of all pairs history, just one specific pair wont work
    Output: Summary statistics for every pair"""
    idx = pd.IndexSlice
    stats = pd.DataFrame(index=df.index.unique(level=0), columns = ['Mean', 'Total profit','Std', 'Sharpe',
     'Number of trades', 'Avg length of position', 'Pct of winning trades', 'Max drawdown'])
    desc = pd.DataFrame(index=['avg'], columns = ['Mean', 'Total profit','Std', 'Sharpe',
     'Number of trades', 'Avg length of position', 'Pct of winning trades', 'Max drawdown'])
    trad = infer_periods(df)['trading']
    tradingdays = abs((trad[0][1]-trad[1][1]).days)
    annualizer = 365/tradingdays
    monthlizer = 30/tradingdays
    riskfree = riskfree/annualizer
    for name, group in df.groupby(level=0):
        stats.loc[name, 'Mean']= group['Profit'].mean()
        stats.loc[name, 'Total profit']=group['Profit'].sum()
        stats.loc[name, 'Std']=group['Profit'].std()
        stats.loc[name, 'Number of trades']=len(group[group['Signals']=='Long'])+len(group[group['Signals']=='Short'])
        #stats.loc[name, 'Roundtrip trades']=(len(group[group['Signals']=='sellLong'])+len(group[group['Signals']=='sellShort']))
        stats.loc[name, 'Roundtrip trades']=(len(group[group['Signals']=='sellLong'])+len(group[group['Signals']=='sellShort']))/max(1,stats.loc[name, 'Number of trades'])
        #stats.loc[name, 'Avg length of position'] = ((len(group[group['Signals']=='keepLong'])/max(len(group[group['Signals']=='Long']),1))+len(group[group['Signals']=='keepShort'])/max(len(group[group['Signals']=='Short']),1))/max(1,stats.loc[name, 'Number of trades'])
        stats.loc[name, 'Avg length of position'] = ((len(group[group['Signals']=='keepLong']))+len(group[group['Signals']=='keepShort']))/max(1,stats.loc[name, 'Number of trades'])
        stats.loc[name, 'Max drawdown'] = abs(drawdown(group).min())
        neg_mask = group['Profit']<0
        stats.loc[name, 'Downside Std'] = group.loc[neg_mask, 'Profit'].std()
        stats.loc[name, 'Sortino'] = (stats.loc[name, 'Total profit']-riskfree)/stats.loc[name, 'Downside Std']
        stats.loc[name, 'Sharpe'] = (stats.loc[name, 'Total profit']-riskfree)/stats.loc[name,'Std']
        stats.loc[name, 'Monthly profit'] = ((stats.loc[name, 'Total profit']+1)**monthlizer)-1
        stats.loc[name, 'Annual profit'] = ((stats.loc[name, 'Total profit']+1)**(annualizer))-1
        if (pd.isna(group['Profit'].quantile(0.05)) == False)&(group['Profit'].quantile(0.05)!=0):
            stats.loc[name, 'VaR']=-(stats.loc[name, 'Total profit']-riskfree)/group['Profit'].quantile(0.05)
        else:
            stats.loc[name, 'VaR']=None
        stats.loc[name, 'Calmar']=(stats.loc[name, 'Annual profit']/stats.loc[name, 'Max drawdown'])
        last_valid = df.loc[idx[name, timeframe[0]:timeframe[1]], 'cumProfit'].last_valid_index()
        #if the pair never trades, then last_valid=None and it would fuck up indexing later
        if last_valid == None:
            last_valid=timeframe[1]
        #we have to make distinction here for the assignment to stats[CumProfit] to work
        #because sometimes we would assign None and sometimes a Series which would give error
        #the sum() is just to convert the Series to a scalar
        if (isinstance(df.loc[idx[name,last_valid], 'cumProfit'], pd.Series)):
            stats.loc[name, 'Cumulative profit']= df.loc[idx[name,last_valid], 'cumProfit'].sum()
        else:
            stats.loc[name, 'Cumulative profit']= 1
        
        #picks the dates on which trades have ended
        mask2 = find_trades(df.loc[idx[name,:], :], timeframe)[1]
        stats.loc[name, 'Pct of winning trades']=(df.loc[idx[name,:], 'cumProfit'][mask2]>1).sum()/max(stats.loc[name, 'Number of trades'],1)
        if nonzero == True:
            stats.loc[stats['Number of trades']== 0, ['Mean', 'Total profit', 'Monthly profit', 'Annual profit']]=None
        if trades_nonzero == True:
            stats.loc[stats['Number of trades']== 0, ['Roundtrip trades', 'Avg length of position']]=None
    return stats

def signals_worker(multidf, timeframe=5, formation=5, threshold=2,lag=0, stoploss=100, num_of_processes=1):
        global enddate
        idx=pd.IndexSlice
        for name, df in multidf.loc[pd.IndexSlice[:, timeframe[0]:timeframe[1]], :].groupby(level=0):
            df['Signals'] = None
            #df.loc[mask,'Signals'] = True
            index=df.index
            #this is technicality because we truncate the DF to just trading period but 
            #in the first few periods the signal generation needs to access prior values
            #which would be None so we just make them adhoc like this
            col = [None for x in range(lag+2)]
            fill = 'None'
            for i in range(len(df)):
                truei=i
                if i-lag<0:
                    col.append(fill)
                    continue
                if (df.loc[index[i-lag], 'normSpread']>stoploss)&(col[i+lag+1] in ['Short', 'keepShort']):
                    fill = 'stopShortLoss'
                    col.append(fill)
                    fill = 'None'
                    continue
                if (df.loc[index[i-lag], 'normSpread']<(-stoploss))&(col[i+lag+1] in ['Long', 'keepLong']):
                    fill = 'stopLongLoss'
                    col.append(fill)
                    fill = 'None'
                    continue
                if (df.loc[index[i-lag],'normSpread']>=threshold)&(df.loc[index[i-lag-1],'normSpread']<threshold)&(col[i+lag-1] not in ["keepShort"])&(col[truei+lag+1] not in ['Short', 'keepShort']):
                    fill = "Short"
                    col.append(fill)
                    fill = "keepShort"
                    continue
                elif (df.loc[index[i-lag],'normSpread']<=0)&(df.loc[index[i-lag-1],'normSpread']>0) & (col[i+lag+1] in ["Short", "keepShort"]):
                    fill = "sellShort"
                    col.append(fill)
                    fill = "None"
                    continue
                elif ((df.loc[index[i-lag],'normSpread']<=(-threshold))&(df.loc[index[i-lag-1],'normSpread']>(-threshold)))& (col[i+lag-1] not in ["keepLong"])&(col[truei+lag+1] not in ['Long', 'keepLong']):
                    #print(i, col, name, col[truei+lag]!='Long', truei)
                    fill = "Long"
                    col.append(fill)
                    fill = "keepLong"
                    continue
                elif (df.loc[index[i-lag],'normSpread']>=0)&(df.loc[index[i-lag-1],'normSpread']<0) & (col[i+lag+1] in ["Long", "keepLong"]):
                    fill = "sellLong"
                    col.append(fill)
                    fill = "None"
                    continue
                col.append(fill)
            col = col[(lag+2):-1]
            col.append("Sell")
            #df['Signals'] = pd.Series(col[1:], index=df.index)
            multidf.loc[pd.IndexSlice[name, timeframe[0]:timeframe[1]], 'Signals'] = pd.Series(col, index=df.index)
        multidf.loc[idx[:, timeframe[1]:enddate], 'Signals']=multidf.loc[idx[:, timeframe[1]:enddate], 'Signals'].fillna(value='pastFormation')
        multidf.loc[idx[:, formation[0]:formation[1]], 'Signals']=multidf.loc[idx[:, formation[0]:formation[1]], 'Signals'].fillna(value='Formation')
        multidf.loc[idx[:, startdate:formation[0]], 'Signals']=multidf.loc[idx[:, startdate:formation[0]], 'Signals'].fillna(value='preFormation')
        #multidf['Signals'] = multidf['Signals'].fillna(value='Formation')
        return multidf
def signals(multidf, timeframe=5, formation = 5, threshold=2, lag = 0, stoploss=100, num_of_processes=1):
    """ Fills in the Signals during timeframe period 
    Outside of the trading period, it fills Formation and pastTrading"""
    if num_of_processes == 1:

        # global enddate
        return signals_worker(multidf, timeframe=timeframe, formation =formation, threshold=threshold, lag=lag)
    if num_of_processes > 1:
        #Those imports are necessary because of the nested structure. it wont be able to access those things from the global namespace so we need to import again
        # global enddate
        # import multiprocess as mp
        if len(multidf.index.unique(level=0))<num_of_processes:
            num_of_processes = len(multidf.index.unique(level=0))
        pool=mp.Pool(num_of_processes, initargs=(enddate,pd))
        split = np.array_split(multidf.index.unique(level=0), num_of_processes)
        split = [multidf.loc[x] for x in split]
        #Im not sure what I was doing here to be honest..
        args_dict = {'trading':timeframe, 'formation':formation, 'threshold':threshold, 'lag':lag, 'stoploss':stoploss, 'num_of_processes':num_of_processes}
        args = [args_dict['trading'], args_dict['formation'], args_dict['threshold'], args_dict['lag'], args_dict['stoploss'], args_dict['num_of_processes']]
        full_args = [[split[i], *args] for i in range(len(split))]
        results = pool.starmap(signals_worker, full_args)
        results=pd.concat(results)
        pool.close()
        pool.join()
        return results
def load_results(name, methods, base='results\\'):
    path = base + name + '\\'
    files = os.listdir(path)
    dfs = []
    for file in files:
        if methods in file:
            df = pd.read_pickle(path+file)
            dfs.append(df)
    return pd.concat(dfs, keys=range(len(dfs)))

def infer_periods(df):
    """Auto detects the Formation and Trading periods
    Works even with MultiIndexed since the periods are the same across all pairs"""
    mask1 = ~((df['Signals']=='Formation')|(df['Signals']=='pastFormation')|(df['Signals']=='preFormation'))
    mask2 = (df['Signals'] == 'Formation')
    trading = (df.index[np.nonzero(mask1)[0][0]], df.index[np.nonzero(mask1)[0][-1]])
    formation =(df.index[np.nonzero(mask2)[0][0]], df.index[np.nonzero(mask2)[0][-1]])
    return {'formation':formation, 'trading':trading}

def descriptive_frame(olddf):
    diag =['Monthly profit', 'Annual profit','Total profit','Std', 'Sharpe','Sortino', 'VaR', 'Calmar',
     'Number of trades', 'Roundtrip trades', 'Avg length of position', 'Pct of winning trades', 'Max drawdown', 'Cumulative profit']
    idx=pd.IndexSlice
    #rebuilds the MultiIndex?
    temp = [[], []]
    for i in range(len(olddf.index.unique(level=0))):
        temp[0].append([i for x in range(len(olddf.loc[i].index.unique(level=0)))])
        temp[1].append([item for item in olddf.loc[i].index.unique(level=0).array])
    temp[0]=[item for sublist in temp[0] for item in sublist]
    temp[1]=[item for sublist in temp[1] for item in sublist]
    df=pd.DataFrame(index=temp, columns=diag)
    #print(df)
    for name, group in df.groupby(level=0):
        test_df = olddf.loc[name].index.unique(level=0)[0]
        stats = descriptive_stats(olddf.loc[name], infer_periods(olddf.loc[(name, test_df)])['trading'])
        for col in df.loc[name].columns:
            df.loc[idx[name, :], col]=stats[col].values

    return df.astype('float32')

def find_same(r1, r2):
    percentages = []
    for i in range(len(r1.index.unique(level=0))):
        same = r1.loc[(i), :].index.unique(level=0)[r1.loc[(i), :].index.unique(level=0).isin(r2.loc[(i), :].index.unique(level=0))]
        percentage = len(same)/len(r1.loc[(i), :].index.unique(level=0))
        percentages.append(percentage)
    return pd.Series(percentages).mean()

def summarize(df, index):
    """ Summarizes the return distribution"""
    res = pd.DataFrame(index=index, columns = [0])
    res.loc['Mean']=df.mean()
    res.loc['Std']=df.std()
    res.loc['Max']=df.max()
    res.loc['Min']=df.min()
    jb = statsmodels.stats.stattools.jarque_beras(df.dropna().values)
    res.loc['Jarque-Bera p-value'] = jb[1]
    res.loc['Kurtosis']=jb[3]
    res.loc['Skewness']=jb[2]
    count = df.count()
    res.loc['Positive']= sum(df>0)/count
    res.loc['t-stat'] = res.loc['Mean']/res.loc['Std']*(count)**(1/2)
    return res

def aggregate(dfs, feasible, freqs=[60,60,10,10], standard=True, returns_nonzero=False, trades_nonzero=False):
    temp = []

    for i in range(len(dfs)):
        df=dfs[i]
        numnom = len(df.index.get_level_values(level=1))/(df.index[-1][0]+1)
        numtr = len(df[df['Number of trades']>0].index.get_level_values(level=1))/(df.index[-1][0]+1)
        if returns_nonzero == True:
            df.loc[df['Number of trades']== 0, ['Total profit', 'Monthly profit', 'Annual profit']]=None
        if trades_nonzero == True:
            df.loc[df['Number of trades']== 0, ['Roundtrip trades', 'Avg length of position']]=None
        mean = df.groupby(level=0).mean()
        mean['Trading period Sharpe'] = (mean['Total profit']-(0.02/(365/freqs[i])))/mean['Std']
        mean['Annualized Sharpe'] = mean['Trading period Sharpe']*((365/freqs[i])**(1/2))
        mean = mean.mean()
        mean['Annual profit'] = ((1+mean['Total profit'])**(365/freqs[i])-1)
        mean['Monthly profit'] =((1+mean['Total profit'])**(30/freqs[i])-1)
        mean['Nominated pairs'] = numnom
        mean['Traded pairs']=numtr
        mean['Traded pairs'] = mean['Traded pairs']/mean['Nominated pairs']
        temp.append(mean[feasible])
    concated = pd.concat(temp, axis=1)
    if standard==True:
        cols = pd.MultiIndex.from_product([['Daily', 
        'Hourly'], ['Distance', 'Cointegration']])
        concated.columns = cols
    return concated

def beautify(df, overlap=False):
    formats={'Monthly profit':"{:.2%}", 'Annual profit':"{:.2%}" ,'Total profit': "{:.2%}", 'Sharpe':"{:.2}", 'Roundtrip trades':"{:.2%}",
    'Number of trades':"{:.3}", 'Avg length of position':"{:3.1f}", 'Pct of winning trades':"{:.2%}",
    'Max drawdown':"{:.2%}", 'Cumulative profit' :"{:.2%}",
    'Mean':"{:1.5f}", 'Std':"{:.3}", 'Max':"{:.3}", 'Min':"{:.3}", 'Jarque-Bera p-value':"{:.3}",
    'Skewness':"{:.3}", 'Kurtosis':"{:.3}", 'Positive':"{:.2%}", 't-stat':"{:.2}",
    'Sortino':"{:.2}", 'Calmar':"{:.2}", 'VaR':"{:.2}", "\% of identical pairs":'{:.2%}',
    'Trading p. Sharpe':"{:.2}", 'Annualized Sharpe':"{:.2}", 'Trading period Sharpe':"{:.2}",
    'Monthly number of trades':"{:.3}", 'Length of position (days)':"{:3.1f}", 'Monthly profit (committed)':"{:.2%}",
    'Nominated pairs':"{:.3}", 'Traded pairs': "{:.2%}"}
    if overlap==False:
        df=df.astype('float32')
        for row in df.index:
            if row in formats.keys():
                df.loc[row]=df.loc[row].map(formats[row].format)
                df.loc[row]=df.loc[row].apply(lambda x: x.replace('%', '\%'))
        df=df.rename(index={"Total profit":'Trading period profit'})
    else:
        df = df.astype('float32')
        for i in range(len(df.index)):
            if df.index[i] in formats.keys():
                df.iloc[i]=df.iloc[i].map(formats[df.index[i]].format)
                df.iloc[i]=df.iloc[i].apply(lambda x: x.replace('%', '\%'))
        df=df.rename(index={"Total profit":'Trading period profit'})
    return df


def rhoci(rho, n, conf=0.95):
    mean=np.arctanh(rho)
    std = 1/((n-3)**(1/2))
    norm = scipy.stats.norm(loc=mean, scale=std)
    ci = [mean-1.96*std, mean+1.96*std]
    trueci = [np.round(np.tanh(ci[0]), 2), np.round(np.tanh(ci[1]), 2)]
    return trueci

def hdist(df):
    grouper = df.index.get_level_values('Time').hour
    res = df.groupby(by=grouper).agg(['mean', 'count', 'std'])
    res['t-stat']=res['mean']/res['std']*res['count'].pow(1/2)
    res.columns = res.columns.str.capitalize()
    res.columns = pd.MultiIndex.from_product([ ['Returns distribution'],res.columns])
    res.index.rename('Hour', inplace=True)
    return res

def stoploss_results(methods = ['dist'], freqs = ['daily'], thresh = ['1','2','3'], stoploss=['2','3','4','5','6']):
    res = {}
    save = 'C:\\Bach\\results\\'
    for f in os.listdir(save):
        if ('dist' in methods) and ('scenarios' in f) and (f[-2] in thresh) and (f[-1] in stoploss) and (f[-3] in [x[0] for x in freqs]):
            res[f]=load_results(f, 'dist')
        if ('coint' in methods) and ('scenarios' in f) and (f[-2] in thresh) and (f[-1] in stoploss) and (f[-3] in [x[0] for x in freqs]):
            res[f]=load_results(f, 'coint')

    return res

def stoploss_preprocess(res, savename, savepath):
    des = {k:descriptive_frame(v) for k,v in res.items()}
    with open(savepath+savename+'.pkl', 'wb') as handle:
        pickle.dump(des, handle, protocol=2)
    pass

def stoploss_table(dict,stem,freq,feasible = ['Monthly profit', 'Annualized Sharpe'], standard=False, thresh=[1,2,3], stoploss=[2,3,4,5,6]):
    rows = []
    for i in range(len(stoploss)):
        temp = []
        for j in range(len(thresh)):
            temp.append(aggregate([dict[stem+str(thresh[j])+str(stoploss[i])]], feasible, standard=standard, freqs=freq, trades_nonzero=True, returns_nonzero=True))
        rows.append(temp)
    return rows
def stoploss_streamed(savename, savepath, methods = ['dist'], freqs = ['daily'], thresh = ['1','2','3'], stoploss=['2','3','4','5','6']):
    des = {}
    save = 'C:\\Bach\\results\\'
    for f in os.listdir(save):
        if ('dist' in methods) and ('scenarios' in f) and (f[-2] in thresh) and (f[-1] in stoploss) and (f[-3] in [x[0] for x in freqs]):
            res=load_results(f, 'dist')
            des[f] = res
            del res
        if ('coint' in methods) and ('scenarios' in f) and (f[-2] in thresh) and (f[-1] in stoploss) and (f[-3] in [x[0] for x in freqs]):
            res=load_results(f, 'coint')
            des[f] = descriptive_frame(res)
            del res
    with open(savepath+savename+'.pkl', 'wb') as handle:
            pickle.dump(des, handle, protocol=pickle.HIGHEST_PROTOCOL)    
        
    return des
def filter_nonsense(df):
    for col in df.columns.get_level_values(level=1):
        
        df.loc[(df.index.get_level_values(level=1)<=col), ('Threshold',col)]='None'
    return df

def produce_stoploss_table(des, prefix, freqs):
    df=stoploss_table(des, prefix, freqs)
    df=pd.concat(list(map(lambda x: pd.concat(x, axis=1), df)))
    df = beautify(df, overlap=True)
    cols = pd.MultiIndex.from_product([['Threshold'], [1,2,3]])
    index = pd.MultiIndex.from_arrays([[2,2,3,3,4,4,5,5,6,6], ['Monthly profit', 'Annualized Sharpe', 'Monthly profit', 'Annualized Sharpe', 'Monthly profit', 'Annualized Sharpe', 'Monthly profit', 'Annualized Sharpe', 'Monthly profit', 'Annualized Sharpe']])
    df.columns = cols
    df.index =index 
    df=pd.concat([df], axis=0, keys=['Stop-loss'])
    df=df.round(3)
    #gets rid of entries where threshold < stoploss
    df = filter_nonsense(df)
    return df

def standardize_results(df, drop=True):
    df.loc['Avg length of position']=df.loc['Avg length of position'].astype('float32')*[1,1,1/24,1/24]
    df.loc['Number of trades']=df.loc['Number of trades'].astype('float32')*[1/2,1/2,3,3]
    df=df.rename({'Avg length of position':'Length of position (days)', 'Number of trades':'Monthly number of trades'})
    if drop==True:
        df=df.drop(['Trading period profit', 'Trading period Sharpe', 'Annual profit', 'Total profit'], errors='ignore')
    return df

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

def aggregate_from_individual_folders():
    os.chdir(r"C:\Bach\test3\1m_data")
    cryptos = os.listdir()
    for crypto in cryptos:
        with cd(crypto):
            shutil.move(crypto + '.csv', r'C:\Bach\test3')


#%%
