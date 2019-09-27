from binance_custom import DataClient
from multiprocess import freeze_support

if __name__ == '__main__':
       freeze_support()
       start='12/31/2017'
       end='9/23/2019'

       pairs=['MDABTC', 'MTLBTC', 'LINKBTC', 'DASHBTC', 'OMGBTC', 'ENJBTC', 'XMRBTC',
              'ARNBTC', 'ZRXBTC', 'QTUMBTC', 'WAVESBTC', 'WTCBTC', 'ETCBTC',
              'IOTABTC', 'XVGBTC', 'LTCBTC', 'ADABTC', 'NEOBTC', 'BNBBTC', 'EOSBTC',
              'XRPBTC', 'TRXBTC', 'ETHBTC']

       pair_list = DataClient().get_binance_pairs(base_currencies=['BTC'])
       pair_list=[x for x in pair_list if x not in pairs]
       store_data = DataClient().kline_data(pair_list,'1m', progress_statements=True, storage = ['csv', 'C:\\Bach\\fresh_admissible_secondwave'])
