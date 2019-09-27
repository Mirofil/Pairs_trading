#%%
from helpers import *
from cointmethod import *
from distancemethod import *
import os
#%%
def simulate(params):
    #freq,lag,txcost,training_delta, cutoff,formation_delta, start, end, jump, methods, dist_num, scenario,truncate=False,redo_x=False, redo_y=False
    freq,lag,txcost,training_delta,cutoff,formation_delta,start,end,jump, methods, dist_num, threshold,stoploss,scenario=params.values()
    redo_x=False
    redo_y=False
    truncate = True
    files = os.listdir(data_folder)
    paths = ['C:\\Bach\\concatenated_price_data\\ '[:-1] + x for x in files if x not in ['BTCUSDT.csv', 'ETHUSDT.csv', 'CLOAKBTC.csv']]
    names = [file.partition('.')[0] for file in files]
    save = 'C:\\Bach\\results\\'
    str_cutoff = str(cutoff).replace('.', '_')
    str_freq = str(freq[1])
    formationdelta = relativedelta(months=formation_delta[0], days=formation_delta[1] ,hours=formation_delta[2])
    trainingdelta = relativedelta(months=training_delta[0], days = training_delta[1], hours = training_delta[2])
    jumpdelta = relativedelta(months=jump[0], days=jump[1], hours=jump[2])
    #5000 is arbirtrarily high limit that will never be reached - but the 
    if not os.path.isdir('C:\\Bach\\results\\'+scenario):
        os.mkdir('C:\\Bach\\results\\'+scenario)
    with open('C:\\Bach\\results\\'+scenario+'\\'+'parameters'+'.txt', 'w') as tf:
        print(params, file=tf)
    for i in range(5000):

        
        formation = (start+i*jumpdelta, start + formationdelta + i*jumpdelta)
        trading = (formation[1], formation[1]+trainingdelta)
        print(formation)
        if (trading[1]>end):
            if truncate == True:
                trading=(trading[0],datetime.date(*[2018,12,31]))
            else:
                break
        if (trading[1]<formation[1]):
            break
        if redo_x == True:
            x=prefilter(paths, cutoff=0.7)
            np.save(save+str(i)+'x'+str(freq)+str(cutoff), x)
            #x=np.load('C:\\Bach\\results\\'+str(i)+'x'+str(freq)+str(cutoff))
        else:
            #x=np.load('C:\\Bach\\results\\'+str(i)+'x'+str(freq)+str(cutoff))
            x=np.load(save+'prefiltered'+str_cutoff+'.npy')
        if redo_y == True:
            y=preprocess(x[:,0], first_n=0, freq=freq)
            y.to_pickle(save+str(i)+'y'+str(freq))
            #y=pd.read_pickle('preprocessed.pkl')
        else:
            y=pd.read_pickle(save + 'preprocessed'+str_freq+ str_cutoff+'.pkl')
        if 'coint' in methods:
            coint_head = pick_range(y, formation[0], formation[1])
            k=cointegration(find_integrated(coint_head))
            coint_spreads = coint_spread(y, [item[0] for item in k], timeframe=formation, betas = [item[1] for item in k])
            coint_spreads.sort_index(inplace=True)
            coint_signal = signals(coint_spreads, timeframe = trading, formation=formation,lag = lag,threshold = threshold, stoploss=stoploss)
            coint_signal = signals_numeric(coint_signal)
            weights_from_signals(coint_signal, cost=txcost)
            propagate_weights(coint_signal, formation)
            calculate_profit(coint_signal, cost=txcost)
            #np.save(save+str(i)+'coint_signal', coint_signal)
            coint_signal.to_pickle(save+scenario+'\\'+str(i)+'coint_signal.pkl')
        if 'dist' in methods:
            head = pick_range(y, formation[0], formation[1])
            distances = distance(head, num = dist_num)
            spreads=distance_spread(y,distances[2], formation)
            spreads.sort_index(inplace=True)
            dist_signal=signals(spreads, timeframe=trading, formation=formation,lag = lag, threshold = threshold, stoploss=stoploss)
            weights_from_signals(dist_signal, cost=txcost)
            propagate_weights(dist_signal, formation)
            calculate_profit(dist_signal, cost=txcost) 
            #np.save(save+scenario+'\\'+str(i)+'dist_signal', dist_signal)
            dist_signal.to_pickle(save+scenario+'\\'+str(i)+'dist_signal.pkl')
            
        if trading[1]==datetime.date(*[2018,12,31]):
            break
        
def stoploss(freqs = ['daily']):
    thresh = [1,2,3]
    stoploss = [2,3,4,5,6]
    save = 'C:\\Bach\\results\\'
    scenariod={"freq":"1D",'lag':1, 'txcost':0.003, 'training_delta':[2,0,0], 'cutoff':0.7, 'formation_delta':[4,0,0], 'start':datetime.date(*[2018,1,1]), 'end':datetime.date(*[2019,1,1]), 'jump':[1,0,0], 'methods':['dist', 'coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':"scenariosd12"}
    scenarioh={"freq":"1H",'lag':1, 'txcost':0.003, 'training_delta':[0,10,0], 'cutoff':0.7, 'formation_delta':[0,20,0], 'start':datetime.date(*[2018,1,1]), 'end':datetime.date(*[2019,1,1]), 'jump':[0,10,0], 'methods':['dist', 'coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':'scenariosh12'}
    if 'daily' in freqs:
        for i in range(len(thresh)):
            for j in range(len(stoploss)):
                newnamed = scenariod['name'][:-2]+str(thresh[i])+str(stoploss[j])
                if os.path.isfile(save+newnamed+'\\'+str(0)+'dist_signal.pkl'):
                    continue
                scenariod.update({'threshold':thresh[i], 'stoploss':stoploss[j], 'name':newnamed})
                simulate(scenariod)
    if 'hourly' in freqs:
        for i in range(len(thresh)):
            for j in range(len(stoploss)):
                newnameh = scenarioh['name'][:-2]+str(thresh[i])+str(stoploss[j])
                if os.path.isfile(save+newnameh+'\\'+str(0)+'coint_signal.pkl'):
                    continue
                scenarioh.update({'threshold':thresh[i], 'stoploss':stoploss[j], 'name':newnameh})
                simulate(scenarioh)        
#%%

