#%%
import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import timeit
import datetime
import scipy
from dateutil.relativedelta import relativedelta
from pairs.distancemethod import *
from pairs.helpers import load_results
from pairs.cointmethod import *
from pairs.config import *
from itertools import zip_longest
from pairs.formatting import beautify, standardize_results
import pickle

from pairs.analysis import descriptive_stats, descriptive_frame, summarize, aggregate

#%%
# DF INTEGRATIONS
# y1=pd.read_pickle('preprocessedD0_0.pkl')
y2 = pd.read_pickle("preprocessedD0_7.pkl")
options = ["c", "ct", "ctt", "nc"]
match = {"c": "$\mu$", "ct": "$\mu + t$", "ctt": "$\mu + t + t^2$", "nc": "neither"}
coint_head = pick_range(y2, formation[0], formation[1])
# results = pd.DataFrame(index=['\# of unit roots (n=81)', '\# of unit roots (n=23)'], columns = ['neither' ,'$\mu$', '$\mu + t$', '$\mu + t + t^2$'])
results = pd.DataFrame(
    index=["\# of unit roots (n=23)"],
    columns=["neither", "$\mu$", "$\mu + t$", "$\mu + t + t^2$"],
)
for option in options:
    results.loc[results.index[0], match[option]] = len(
        find_integrated(y2, regression=option).index.unique(level=0)
    )
    # results.loc[results.index[1] ,match[option]]=len(find_integrated(y2, regression=option).index.unique(level=0))
latexsave(results, save_path_tables + "unitroots")
#%%
# LIST OF PAIRS
y1 = pd.read_pickle("preprocessedD0_0.pkl")
y2 = pd.read_pickle("preprocessedD0_7.pkl")
files = os.listdir(data_path)
# we exclude CLOAKBTC because theres some data-level mixed types mistake that breaks prefilter and it would get deleted anyways
# it also breakts at ETHBTC (I manually deleted the first wrong part in Excel)
paths = [
    data_path + x
    for x in files
    if x not in ["BTCUSDT.csv", "ETHUSDT.csv", "CLOAKBTC.csv"]
]
listnames = [file.partition(".")[0] for file in files]
listnames = [name[:-3] for name in listnames]
# counterparts = ['Cardano', 'AdEx', 'Aeternity', 'SingularityNet', 'Aion', 'Ambrosus',
# 'AppCoins', 'Ardor', 'Ark', 'Aeron', 'AirSwap', 'Basic Attention Token',
# 'BitConnect', 'Bitcoin Diamond' ,'Bitcoin Cash ABC', 'Bitcoin SV', 'Bytecoin', 'BlockMason',
# 'Bluzelle', 'Binance Coin', 'Bancor', 'Ethos', 'Bread', 'BitcoinUltra', 'Bitcoin Gold',
# 'BitShares', 'BitTorrent','Blox', 'ChatCoin', 'CloakCoin', 'CyberMiles', 'Cindicator',
# 'Civic', 'Dash', 'Data', 'Decred', 'Dent','DigixDAO', 'Agrello', 'district0x','Dock', 'Eidoo', 'aelf', 'Enigma', 'EnjinCoin',
# 'EOS', 'Ethereum Classic', 'Ethereum', 'Ethereum Unlimited', 'Everex', 'Fetch',
# 'Etherparty', 'FunFair', 'Gas', 'Golem', 'GoChain', 'Groestlcoin', 'Gifto', 'Genesis Vision',
# 'GXChain', 'HyperCash', 'Holo', 'HyperCash', 'Iconomi', 'ICON', 'Insolar', 'IOST', 'Iota', 'IoTeX',
# 'Selfkey', 'Komodo', 'Kyber Network', 'ETHLend', 'Chainlink', 'Loom Network',
# 'Loopring', 'Lisk', 'LiteCoin', 'Lunyr', 'Decentraland', 'Monaco', 'Moeda', 'Mainframe', 'Mithril','Modum','Monetha',
# 'Metal', 'Nano', 'Nebulas', 'NavCoin', 'Nucleus Vision', 'Neblio', 'NEO', 'Pundi X',
# 'NULS', 'Nexus', 'OAX', 'OmiseGO', 'Ontology Gas', 'Ontology', 'OST', 'Paxos',
# 'Red Pulse Phoenix', 'PIVX', 'POA Network', 'Po.et', 'Polymath', 'Power Ledger',
# 'Populous', 'QuarkChain', 'QLC Chain', 'Quantstamp', 'Qtum', 'Rcoin', 'Raiden Network',
# 'Ren', 'Augur', 'Request', 'iExec', 'Red Pulse Phoenix', 'Ravencoin', 'SALT', 'Siacoin',
# 'Skycoin', 'SingularDTV', 'SONM', 'Status', 'Steem', 'Storj', 'Storm', 'Stratis',
# 'Substratum', 'Syscoin', 'Theta', 'Time New Bank', 'Tierion', 'Triggers',
# 'Tron', 'TrueUSD', 'USD Coin', 'Vechain', 'VeChain', 'Viacoin', 'Viberate','VIBE',
# 'Tael', 'Wanchain', 'Waves', 'Wings', 'WePower', 'Waltonchain', 'NEM', 'Stellar',
# 'Monero', 'Ripple', 'Verge', 'Zcoin', 'YOYOW', 'Zcash', 'Horizen', 'Zilliqa', '0x']
# matched = dict(zip_longest(listnames, counterparts))
# matcheddf= pd.DataFrame.from_dict(matched, orient='index')
mask1 = matcheddf.index.isin([item[:-3] for item in list(y1.index.unique(level=0))])
mask2 = matcheddf.index.isin([item[:-3] for item in list(y2.index.unique(level=0))])
# matcheddf['No Cutoff'] = pd.Series(mask1).values
# matcheddf['Cutoff'] = pd.Series(mask2).values
# matcheddf.replace(False, 'Not incl.', inplace=True)
# matcheddf.replace(True, 'Incl.', inplace=True)
# latexsave(matcheddf, save_path_tables+'cryptolist')

#%%
# SAME PAIRS BETWEEN METHODS
r_dist_daily = load_results("scenario1", "dist")
r_dist_hourly = load_results("scenario3", "dist")
r_coint_daily = load_results("scenario1", "coint")
r_coint_hourly = load_results("scenario3_nolag", "coint")
r_dist_daily = load_results("scenario1", "dist")
r_dist_hourly = load_results("scenario3", "dist")
r_coint_daily = load_results("scenario1", "coint")
r_coint_hourly = load_results("scenario3_nolag", "coint")
index1 = ["\% of identical pairs"]
cutoffs = ["Vol. cutoff", "No cutoff"]
multiindex = pd.MultiIndex.from_product([index1, cutoffs])
table = pd.DataFrame(index=index1, columns=["daily", "hourly"])
table.loc["\% of identical pairs", "hourly"] = find_same(r_dist_hourly, r_coint_hourly)
table.loc["\% of identical pairs", "daily"] = find_same(r_dist_daily, r_coint_daily)
table.loc["\% of identical pairs"] = table.loc["\% of identical pairs"]
multitable = pd.DataFrame(index=multiindex, columns=["daily", "hourly"])
multitable.loc[("\% of identical pairs", "Vol. cutoff"), "hourly"] = find_same(
    r_dist_hourly, r_coint_hourly
)
multitable.loc[("\% of identical pairs", "Vol. cutoff"), "daily"] = find_same(
    r_dist_daily, r_coint_daily
)

table = beautify(table)
multitable = beautify(multitable)
# have to finish multitable with No Cutoff but have no data atm


latexsave(table, save_path_tables + "identical")
latexsave(multitable, save_path_tables + "identical_multi")
#%%
# CORRELATIONS
r1 = load_results("scenario1", "dist")
tabledc = load_results("scenario1", "coint")
r3 = load_results("scenario3", "dist")
r4 = load_results("scenario3", "coint")
de1 = descriptive_frame(r1)
de2 = descriptive_frame(tabledc)
de3 = descriptive_frame(r3)
de4 = descriptive_frame(r4)
corr1 = corrs(de1)
corr2 = corrs(de2)
corr3 = corrs(de3)
corr4 = corrs(de4)
corr = pd.concat([corr1[0], corr2[0]], axis=1, keys=["Distance", "Coint"])
corr_appendix = pd.concat([corr3[0], corr4[0]], axis=1, keys=["Distance", "Coint"])

cis1 = pd.concat([corr1[1], corr2[1]], axis=0, keys=["Distance", "Coint"])
cis2 = pd.concat([corr3[1], corr4[1]], axis=0, keys=["Distance", "Coint"])
cis = pd.concat([cis1, cis2], axis=0, keys=["Daily", "Hourly"])

corr = beautify(corr)
corr_appendix = beautify(corr_appendix)

latexsave(corr, save_path_tables + "distcorrs")
latexsave(corr_appendix, save_path_tables + "distcorrs_hourly")
latexsave(cis, save_path_tables + "corrcis")
#%%
# RETURNS NORMALITY?
newbase = "NEWresults\\"
rdd = load_results("scenario1", "dist", newbase)
rhd = load_results("scenario3", "dist", newbase)
rdc = load_results("scenario1", "coint", newbase)
rhc = load_results("scenario3", "coint", newbase)
rtd = load_results("scenario7", "dist", newbase)
rtc = load_results("scenario7", "coint", newbase)
retdd = rdd["Profit"]
rethd = rhd["Profit"]
retdc = rdc["Profit"]
rethc = rhc["Profit"]
rettd = rtd["Profit"]
rettc = rtc["Profit"]


#%%
index = [
    "Mean",
    "Std",
    "Max",
    "Min",
    "Jarque-Bera p-value",
    "Skewness",
    "Kurtosis",
    "Positive",
    "t-stat",
]
dailyn = round((retdd.count() + retdc.count()) / 2)
hourlyn = round((rethd.count() + rethc.count()) / 2)
cols = pd.MultiIndex.from_product(
    [["Daily", "Hourly", "5-Minute"], ["Dist.", "Coint."]]
)
# figure out how to put n values in the headers
df = pd.DataFrame(index=index, columns=cols)
df["Daily", "Dist."] = summarize(retdd, index).values
df["Daily", "Coint."] = summarize(retdc, index).values
df["Hourly", "Dist."] = summarize(rethd, index).values
df["Hourly", "Coint."] = summarize(rethc, index).values
df["5-Minute", "Dist."] = summarize(rettd, index).values
df["5-Minute", "Coint."] = summarize(rettc, index).values
df.to_pickle(save_path_tables + "retstats.pkl")
df.round(4)
df = beautify(df)
latexsave(df, save_path_tables + "retdist")

#%%
# GIANT RESULT TABLE
newbase = "paper1/NEWresults"
rdd = load_results("scenario1", "dist", newbase)
rhd = load_results("scenario3", "dist", newbase)
rdc = load_results("scenario1", "coint", newbase)
rhc = load_results("scenario3", "coint", newbase)
rtd = load_results("scenario7", "dist", newbase)
rtc = load_results("scenario7", "coint", newbase)
ddd = descriptive_frame(rdd)
dhd = descriptive_frame(rhd)
ddc = descriptive_frame(rdc)
dhc = descriptive_frame(rhc)
print("Everything but minute scenarios done")
dtd = descriptive_frame(rtd)
dtc = descriptive_frame(rtc)
ddd.to_pickle(save_path_tables + "ddd.pkl")
dhd.to_pickle(save_path_tables + "dhd.pkl")
ddc.to_pickle(save_path_tables + "ddc.pkl")
dhc.to_pickle(save_path_tables + "dhc.pkl")
dtd.to_pickle(save_path_tables + "dtd.pkl")
dtc.to_pickle(save_path_tables + "dtc.pkl")
ddd = pd.read_pickle(save_path_tables + "ddd.pkl")
dhd = pd.read_pickle(save_path_tables + "dhd.pkl")
dhc = pd.read_pickle(save_path_tables + "dhc.pkl")
ddc = pd.read_pickle(save_path_tables + "ddc.pkl")
dtd = pd.read_pickle(save_path_tables + "dtd.pkl")
dtc = pd.read_pickle(save_path_tables + "dtc.pkl")

#%%
feasible = [
    "Monthly profit",
    "Annual profit",
    "Total profit",
    "Annualized Sharpe",
    "Trading period Sharpe",
    "Number of trades",
    "Roundtrip trades",
    "Avg length of position",
    "Pct of winning trades",
    "Max drawdown",
    "Nominated pairs",
    "Traded pairs",
]
agg = aggregate(
    [ddd, ddc, dhd, dhc], columns_to_pick=feasible, multiindex_from_product_cols=[["Daily", "Hourly"], ["Dist.", "Coint."]], trades_nonzero=True, returns_nonzero=True
)
agg = standardize_results(agg)
agg = beautify(agg)
latexsave(agg, save_path_tables + "resultstable")
#%%
# 5MINUTE AGGREGATION
feasible = [
    "Monthly profit",
    "Annual profit",
    "Total profit",
    "Annualized Sharpe",
    "Trading period Sharpe",
    "Number of trades",
    "Roundtrip trades",
    "Avg length of position",
    "Pct of winning trades",
    "Max drawdown",
    "Nominated pairs",
    "Traded pairs",
]
agg = aggregate(
    [ddd, ddc, dhd, dhc, dtd, dtc],
    columns_to_pick=feasible,
    trades_nonzero=True,
    returns_nonzero=True,
    trading_period_days=[60, 60, 10, 10, 3, 3],
)
agg = standardize_results(
    agg,
    poslen=[1, 1, 1 / 24, 1 / 24, 1 / 288, 1 / 288],
    numtrades=[1 / 2, 1 / 2, 3, 3, 10, 10],
)
agg = beautify(agg)
latexsave(agg, save_path_tables + "MINUTEresultstable")
#%%
# NOLAG TABLE
newbase = "NEWresults\\"
rhdnl = load_results("scenario3_nolag", "dist", newbase)
rddnl = load_results("scenario1_nolag", "dist", newbase)
rdcnl = load_results("scenario1_nolag", "coint", newbase)
rhcnl = load_results("scenario3_nolag", "coint", newbase)
rtdnl = load_results("scenario7_nolag", "dist", newbase)
rtcnl = load_results("scenario7_nolag", "coint", newbase)
dddnl = descriptive_frame(rddnl)
dhdnl = descriptive_frame(rhdnl)
ddcnl = descriptive_frame(rdcnl)
dhcnl = descriptive_frame(rhcnl)
print("First part done")
dtdnl = descriptive_frame(rtdnl)
dtcnl = descriptive_frame(rtcnl)
dddnl.to_pickle(save_path_tables + "dddnl.pkl")
dhdnl.to_pickle(save_path_tables + "dhdnl.pkl")
ddcnl.to_pickle(save_path_tables + "ddcnl.pkl")
dhcnl.to_pickle(save_path_tables + "dhcnl.pkl")
dtdnl.to_pickle(save_path_tables + "dtdnl.pkl")
dtcnl.to_pickle(save_path_tables + "dtcnl.pkl")
dddnl = pd.read_pickle(save_path_tables + "dddnl.pkl")
dhdnl = pd.read_pickle(save_path_tables + "dhdnl.pkl")
dhcnl = pd.read_pickle(save_path_tables + "dhcnl.pkl")
ddcnl = pd.read_pickle(save_path_tables + "ddcnl.pkl")
dtdnl = pd.read_pickle(save_path_tables + "dtdnl.pkl")
dtcnl = pd.read_pickle(save_path_tables + "dtcnl.pkl")

#%%
feasible = [
    "Monthly profit",
    "Annual profit",
    "Total profit",
    "Annualized Sharpe",
    "Trading period Sharpe",
    "Number of trades",
    "Roundtrip trades",
    "Avg length of position",
    "Pct of winning trades",
    "Max drawdown",
]
# careful about ordering of commited/employed because it changes xxxnl inplace
aggnl_commited = aggregate([dddnl, ddcnl, dhdnl, dhcnl], feasible, trades_nonzero=True)
aggnl = aggregate(
    [dddnl, ddcnl, dhdnl, dhcnl],
    feasible,
    trades_nonzero=True,
    returns_nonzero=True,
    trading_period_days=[60, 60, 10, 10, 3, 3],
)
aggnl.loc["Monthly profit (committed)"] = aggnl_commited.loc["Monthly profit"]
aggnl = standardize_results(aggnl)
# standardize results has to be first because it renames some rows
aggnl = aggnl.loc[
    [
        "Monthly profit",
        "Monthly profit (committed)",
        "Annualized Sharpe",
        "Monthly number of trades",
        "Roundtrip trades",
        "Length of position (days)",
        "Pct of winning trades",
        "Max drawdown",
    ]
]
aggnl = beautify(aggnl)
latexsave(aggnl, save_path_tables + "resultstablenolag")

# 5 MINUTE
feasible = [
    "Monthly profit",
    "Annual profit",
    "Total profit",
    "Annualized Sharpe",
    "Trading period Sharpe",
    "Number of trades",
    "Roundtrip trades",
    "Avg length of position",
    "Pct of winning trades",
    "Max drawdown",
]
# careful about ordering of commited/employed because it changes xxxnl inplace
aggnl_commited = aggregate(
    [dddnl, ddcnl, dhdnl, dhcnl, dtdnl, dtcnl],
    feasible,
    trades_nonzero=True,
    trading_period_days=[60, 60, 10, 10, 3, 3],
)
aggnl = aggregate(
    [dddnl, ddcnl, dhdnl, dhcnl, dtdnl, dtcnl],
    feasible,
    trades_nonzero=True,
    returns_nonzero=True,
    trading_period_days=[60, 60, 10, 10, 3, 3],
)
aggnl.loc["Monthly profit (committed)"] = aggnl_commited.loc["Monthly profit"]
aggnl = standardize_results(
    aggnl,
    poslen=[1, 1, 1 / 24, 1 / 24, 1 / 288, 1 / 288],
    numtrades=[1 / 2, 1 / 2, 3, 3, 10, 10],
)
# standardize results has to be first because it renames some rows
aggnl = aggnl.loc[
    [
        "Monthly profit",
        "Monthly profit (committed)",
        "Annualized Sharpe",
        "Monthly number of trades",
        "Roundtrip trades",
        "Length of position (days)",
        "Pct of winning trades",
        "Max drawdown",
    ]
]
aggnl = beautify(aggnl)
latexsave(aggnl, save_path_tables + "MINUTEresultstablenolag")
# feasible = ['Monthly profit', 'Annual profit' ,'Total profit',  'Annualized Sharpe','Trading period Sharpe', 'Number of trades', 'Roundtrip trades',
# 'Avg length of position', 'Pct of winning trades', 'Max drawdown']
# aggnl_commited = aggregate([dddnl, ddcnl, dhdnl, dhcnl], feasible, trades_nonzero=True)

# aggnl_commited=standardize_results(aggnl_commited)
# aggnl_commited=beautify(aggnl_commited)
# latexsave(aggnl_commited, save_path_tables+'resultstablenolagcommited')
#%%
# NO tx table
newbase = "NEWresults\\"
rhdtx = load_results("scenario4", "dist", newbase)
rddtx = load_results("scenario2", "dist", newbase)
rdctx = load_results("scenario2", "coint", newbase)
rhctx = load_results("scenario4", "coint", newbase)
rtdtx = load_results("scenario8", "dist", newbase)
rtctx = load_results("scenario8", "coint", newbase)
dddtx = descriptive_frame(rddtx)
dhdtx = descriptive_frame(rhdtx)
ddctx = descriptive_frame(rdctx)
dhctx = descriptive_frame(rhctx)
dtdtx = descriptive_frame(rtdtx)
dtctx = descriptive_frame(rtctx)
dddtx.to_pickle(save_path_tables + "dddtx.pkl")
dhdtx.to_pickle(save_path_tables + "dhdtx.pkl")
ddctx.to_pickle(save_path_tables + "ddctx.pkl")
dhctx.to_pickle(save_path_tables + "dhctx.pkl")
dtdtx.to_pickle(save_path_tables + "dtdtx.pkl")
dtctx.to_pickle(save_path_tables + "dtctx.pkl")
dddtx = pd.read_pickle(save_path_tables + "dddtx.pkl")
dhdtx = pd.read_pickle(save_path_tables + "dhdtx.pkl")
dhctx = pd.read_pickle(save_path_tables + "dhctx.pkl")
ddctx = pd.read_pickle(save_path_tables + "ddctx.pkl")

#%%
feasible = [
    "Monthly profit",
    "Annual profit",
    "Total profit",
    "Annualized Sharpe",
    "Trading period Sharpe",
    "Number of trades",
    "Roundtrip trades",
    "Avg length of position",
    "Pct of winning trades",
    "Max drawdown",
]
aggtx = aggregate(
    [dddtx, ddctx, dhdtx, dhctx], feasible, trades_nonzero=True, returns_nonzero=True
)
aggtx = standardize_results(aggtx)
aggtx = beautify(aggtx)
latexsave(aggtx, save_path_tables + "resultstablenotx")

# 5MINUTE
feasible = [
    "Monthly profit",
    "Annual profit",
    "Total profit",
    "Annualized Sharpe",
    "Trading period Sharpe",
    "Number of trades",
    "Roundtrip trades",
    "Avg length of position",
    "Pct of winning trades",
    "Max drawdown",
]
aggtx = aggregate(
    [dddtx, ddctx, dhdtx, dhctx, dtdtx, dtctx],
    feasible,
    trades_nonzero=True,
    returns_nonzero=True,
    trading_period_days=[60, 60, 10, 10, 3, 3],
)
aggtx = standardize_results(
    aggtx,
    poslen=[1, 1, 1 / 24, 1 / 24, 1 / 288, 1 / 288],
    numtrades=[1 / 2, 1 / 2, 3, 3, 10, 10],
)
aggtx = beautify(aggtx)
latexsave(aggtx, save_path_tables + "MINUTEresultstablenotx")
#%%
# Hourly returns distribution table
# formatting is hell of converting to string and back
hdisthd = hdist(rethd.astype("float32"))
hdisthc = hdist(rethc.astype("float32"))
hdisttable = pd.concat([hdisthd, hdisthc], keys=["Distance", "Cointegration"], axis=1)
hdisttable = hdisttable.astype("float32")
for col in hdisttable.columns:
    if col[2] not in ["Count"]:
        hdisttable[col] = hdisttable[col].map("{:.2g}".format)
    if col[2] in ["Count"]:
        hdisttable[col] = hdisttable[col].map("{:.4g}".format)
        hdisttable[col] = hdisttable[col].apply(lambda x: int(float(x)))
    if col[2] in ["Mean"]:
        hdisttable[col] = hdisttable[col].astype("float32")
        hdisttable[col] = hdisttable[col].map("{:.5f}".format)

latexsave(hdisttable, save_path_tables + "hdist")


#%%
# BIG STOPLOSS TABLE
newbase = "NEWresults" + os.sep
resdd = stoploss_results(newbase, methods=["dist"], freqs=["daily"])
resdc = stoploss_results(newbase, methods=["coint"], freqs=["daily"])
reshd = stoploss_results(newbase="NEWresults", methods=["dist"], freqs=["hourly"])
reshc = stoploss_results(newbase="NEWresults", methods=["coint"], freqs=["hourly"])
#%%

stoploss_preprocess(resdd, "desdd", save_path_tables)
stoploss_preprocess(resdc, "desdc", save_path_tables)
stoploss_preprocess(reshd, "deshd", save_path_tables)
stoploss_preprocess(reshc, "deshc", save_path_tables)

#%%
desdd = pickle.load(open(save_path_tables + "desdd.pkl", "rb"))
desdc = pickle.load(open(save_path_tables + "desdc.pkl", "rb"))
deshd = pickle.load(open(save_path_tables + "deshd.pkl", "rb"))
deshc = pickle.load(open(save_path_tables + "deshc.pkl", "rb"))

tabledd = produce_stoploss_table(desdd, "scenariosd", [60])
latexsave(tabledd, save_path_tables + "stoplossdd")

tabledc = produce_stoploss_table(desdc, "scenariosd", [60])
latexsave(tabledc, save_path_tables + "stoplossdc")


tablehd = produce_stoploss_table(deshd, "scenariosh", [10])
latexsave(tablehd, save_path_tables + "stoplosshd")

tablehc = produce_stoploss_table(deshc, "scenariosh", [10])
latexsave(tablehc, save_path_tables + "stoplosshc")

concath = pd.concat([tablehd, tablehc], keys=["Distance", "Cointegration"], axis=1)
concatd = pd.concat([tabledd, tabledc], keys=["Distance", "Cointegration"], axis=1)
latexsave(concath, save_path_tables + "stoplossh")
latexsave(concatd, save_path_tables + "stoplossd")
#%%

