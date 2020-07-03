import datetime
from config import *
#Order sensitive!

#DAILY
starting_date = startdate
ending_date = enddate

scenario_random={"freq":"1D",'lag':1, 'txcost':0.003, 'training_delta':[2,0,0], 'cutoff':0.7, 'formation_delta':[4,0,0], 'start':starting_date, 'end':ending_date, 'jump':[1,0,0], 'methods':['random'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':"scenario_random"}

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
scenario_randomh={"freq":"1H",'lag':1, 'txcost':0.003, 'training_delta':[0,10,0], 'cutoff':0.7, 'formation_delta':[0,20,0], 'start':starting_date, 'end':ending_date, 'jump':[0,10,0], 'methods':['random'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':'scenario_randomh'}

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

#MINUTE
scenario5={"freq":"1T",'lag':1, 'txcost':0.003, 'training_delta':[0,1,0], 'cutoff':0.7, 'formation_delta':[0,2,0], 'start':starting_date, 'end':ending_date, 'jump':[0,2,0], 'methods':['dist'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':"scenario5"}
scenario6={"freq":"1T",'lag':1, 'txcost':0.000, 'training_delta':[0,1,0], 'cutoff':0.7, 'formation_delta':[0,2,0], 'start':starting_date, 'end':ending_date, 'jump':[0,2,0], 'methods':['dist'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':'scenario6'}
scenario5_coint={"freq":"1T",'lag':1, 'txcost':0.003, 'training_delta':[0,1,0], 'cutoff':0.7, 'formation_delta':[0,2,0], 'start':starting_date, 'end':ending_date, 'jump':[0,2,0], 'methods':['coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':"scenario5"}
scenario6_coint={"freq":"1T",'lag':1, 'txcost':0.000, 'training_delta':[0,1,0], 'cutoff':0.7, 'formation_delta':[0,2,0], 'start':starting_date, 'end':ending_date, 'jump':[0,2,0], 'methods':['coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':'scenario6'}

scenario5_coint_nolag={"freq":"1T",'lag':0, 'txcost':0.003, 'training_delta':[0,1,0], 'cutoff':0.7, 'formation_delta':[0,2,0], 'start':starting_date, 'end':ending_date, 'jump':[0,2,0], 'methods':['coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':"scenario5_nolag"}
scenario5_nolag={"freq":"1T",'lag':0, 'txcost':0.003, 'training_delta':[0,1,0], 'cutoff':0.7, 'formation_delta':[0,2,0], 'start':starting_date, 'end':ending_date, 'jump':[0,2,0], 'methods':['dist'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':"scenario5_nolag"}


#5MINUTE
scenario_randomt={"freq":"5T",'lag':1, 'txcost':0.003, 'training_delta':[0,3,0], 'cutoff':0.7, 'formation_delta':[0,6,0], 'start':starting_date, 'end':ending_date, 'jump':[0,6,0], 'methods':['random'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':"scenario_randomt"}
scenario_randomt_full_coverage = {"freq":"5T",'lag':1, 'txcost':0.003, 'training_delta':[0,3,0], 'cutoff':0.7, 'formation_delta':[0,6,0], 'start':starting_date, 'end':ending_date, 'jump':[0,3,0], 'methods':['random'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':"scenario_randomt_full_coverage"}

scenario7={"freq":"5T",'lag':1, 'txcost':0.003, 'training_delta':[0,3,0], 'cutoff':0.7, 'formation_delta':[0,6,0], 'start':starting_date, 'end':ending_date, 'jump':[0,6,0], 'methods':['dist'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':"scenario7"}
scenario8={"freq":"5T",'lag':1, 'txcost':0.000, 'training_delta':[0,3,0], 'cutoff':0.7, 'formation_delta':[0,6,0], 'start':starting_date, 'end':ending_date, 'jump':[0,6,0], 'methods':['dist'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':'scenario8'}
scenario7_coint={"freq":"5T",'lag':1, 'txcost':0.003, 'training_delta':[0,3,0], 'cutoff':0.7, 'formation_delta':[0,6,0], 'start':starting_date, 'end':ending_date, 'jump':[0,6,0], 'methods':['coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':"scenario7"}
scenario8_coint={"freq":"5T",'lag':1, 'txcost':0.000, 'training_delta':[0,3,0], 'cutoff':0.7, 'formation_delta':[0,6,0], 'start':starting_date, 'end':ending_date, 'jump':[0,6,0], 'methods':['coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':'scenario8'}

scenario7_coint_nolag={"freq":"5T",'lag':0, 'txcost':0.003, 'training_delta':[0,3,0], 'cutoff':0.7, 'formation_delta':[0,6,0], 'start':starting_date, 'end':ending_date, 'jump':[0,6,0], 'methods':['coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':"scenario7_nolag"}
scenario7_nolag={"freq":"5T",'lag':0, 'txcost':0.003, 'training_delta':[0,3,0], 'cutoff':0.7, 'formation_delta':[0,6,0], 'start':starting_date, 'end':ending_date, 'jump':[0,6,0], 'methods':['dist'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':"scenario7_nolag"}

scenario7_full_coverage = {"freq":"5T",'lag':1, 'txcost':0.003, 'training_delta':[0,3,0], 'cutoff':0.7, 'formation_delta':[0,6,0], 'start':starting_date, 'end':ending_date, 'jump':[0,3,0], 'methods':['dist'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':"scenario7_full_coverage"}
scenario7_coint_full_coverage = {"freq":"5T",'lag':1, 'txcost':0.003, 'training_delta':[0,3,0], 'cutoff':0.7, 'formation_delta':[0,6,0], 'start':starting_date, 'end':ending_date, 'jump':[0,3,0], 'methods':['coint'], 'dist_num':20, 'threshold':2, 'stoploss':100,'name':"scenario7_full_coverage"}