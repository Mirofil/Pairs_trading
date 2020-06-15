#### you can also download the stock lists from here https://old.nasdaq.com/screening/company-list.aspx

import pandas as pd
import yfinance as yf
import os, contextlib
import shutil
from os.path import isfile, join
from tqdm import tqdm

if __name__ == "__main__":
    offset = 0
    limit = 30000
    period = 'max' # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max

    os.mkdir('hist')

    nasdaq = pd.read_csv("stock_lists/companylist_nasdaq.csv")
    nyse = pd.read_csv("stock_lists/companylist_nyse.csv")
    amex = pd.read_csv("stock_lists/companylist_amex.csv")

    data = pd.read_csv("http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt", sep='|')
    data_clean = data[data['Test Issue'] == 'N']
    symbols = data_clean['NASDAQ Symbol'].tolist()
    print('total number of symbols traded = {}'.format(len(symbols)))


    all_symbols = {'nasdaq': nasdaq["Symbol"], 'nyse': nyse["Symbol"], 'amex':amex["Symbol"]}
    failures = {'nasdaq':[], 'amex':[], 'nyse':[]}
    for exchange, symbols in all_symbols.items():
        os.mkdir(f'hist/{exchange}')

        limit = limit if limit else len(symbols)
        end = min(offset + limit, len(symbols))
        is_valid = [False] * len(symbols)
        # force silencing of verbose API
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                for i in tqdm(range(offset, end), total=end-offset, desc=f'Downloading {exchange} stocks'):
                    try:
                        s = symbols[i]
                        data = yf.download(s, period=period)
                        if len(data.index) == 0:
                            continue
                    
                        is_valid[i] = True
                        data.to_csv(f'hist/{exchange}/{s}.csv')
                    except:
                        print(f"Failed at {s} in {exchange}")
                        failures[exchange].append(s)

        print('Total number of valid symbols downloaded = {}'.format(sum(is_valid)))

        # valid_data = data_clean[is_valid]
        # valid_data.to_csv('symbols_valid_meta.csv', index=False)

        # etfs = valid_data[valid_data['ETF'] == 'Y']['NASDAQ Symbol'].tolist()
        # stocks = valid_data[valid_data['ETF'] == 'N']['NASDAQ Symbol'].tolist()


        # def move_symbols(symbols, dest):
        #     for s in symbols:
        #         filename = '{}.csv'.format(s)
        #         shutil.move(join('hist', filename), join(dest, filename))
                
        # move_symbols(etfs, "etfs")
        # move_symbols(stocks, "stocks")

    # os.rmdir("hist")