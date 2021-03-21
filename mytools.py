import MetaTrader5 as mt5
from datetime import datetime
import pandas as pd


def historyData(Symbol):
    mt5.initialize(server="ForexClub-MT5 Demo Server",login=500063649,password="hrOmcAAn")
    # print(mt5.terminal_info())
    # print(mt5.version())
    # listSymbols=mt5.symbols_get()
    # [x.name for x in listSymbols]
    # Symbol=np.random.choice(FXmajor, 1)[0]
    # print(Symbol)
    # pointValue=mt5.symbol_info(Symbol).point
    # mt5.Buy("EURUSD", 0.1,price=11395,ticket=9)
    # Num_velas=10000
    # Copying data to pandas data frame
    # rates =  mt5.copy_rates_from_pos(Symbol, mt5.TIMEFRAME_M1, 0, Num_velas)
    rates =  mt5.copy_rates_range(Symbol, mt5.TIMEFRAME_M15, datetime(2021, 1, 1), datetime.now())
    # rates =  mt5.copy_rates_range("ES", mt5.TIMEFRAME_D1, datetime(2019, 1, 15), datetime(2019, 1, 25))
    # rates =  mt5.copy_rates_from_pos(Symbol, mt5.TIMEFRAME_M1, 0, Num_velas)

    # Deinitializing MT5 connection
    mt5.shutdown()
    # create DataFrame out of the obtained data
    rates_frame = pd.DataFrame(rates)
    # convert time in seconds into the datetime format
    rates_frame.index=pd.to_datetime(rates_frame['time'], unit='s')

    rates_frame.columns=['time', 'Open', 'High', 'Low', 'Close', 'tick_volume', 'spread','real_volume']
    return rates_frame[['Open', 'High', 'Low', 'Close', 'tick_volume', 'spread']]