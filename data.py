from db import Db
import pendulum
import numpy as np
import pandas as pd

def main():
    db = Db()
    start = pendulum.datetime(2018, 2, 1)
    # end = pendulum.datetime(2019, 1, 1)
    end = pendulum.datetime(2018, 12, 31)
    result_set = [] 
    symbols = ['BTCUSD', 'XRPUSD', 'XRPBTC']
    while start < end:
        key = start.to_date_string()
        result = [key]

        # Input
        sma = db.get_simple_moving_avg([5, 10, 30], start, symbols)
        btc_sent = sum([i[2] for i in np.genfromtxt(f'sentimental/bitcoin/{start.to_date_string()}.csv', delimiter=',')])
        xrp_sent = sum([i[2] for i in np.genfromtxt(f'sentimental/ripple/{start.to_date_string()}.csv', delimiter=',')])
        result.extend(sma)
        result.extend([btc_sent, xrp_sent])

        # Target
        today_close, tmr_close = db.get_today_tmr_close(start, symbols)
        target = get_target(today_close, tmr_close)
        result.append(target)

        # Save
        result_set.append(result)

        start = start.add(days=1)

    temp = np.asarray(result_set)
    pd.DataFrame(temp).to_csv("vozw.csv", index=None)
    
def get_target(today_close, tmr_close):
    # [1,0,0] -> buy btc
    # [0,1,0] -> buy xrp
    # [0,0,1] -> buy usd
    # or just 0, 1, 2
    td = np.array(today_close)
    tmr = np.array(tmr_close)
    delta = tmr - td
    delta_pct = delta / td
    btc, xrp, xrp_btc = delta_pct
    if btc < 0 and xrp < 0:

        return 2 # buy usd
    if xrp_btc > 0:
        return 1 # buy xrp

    return 0 # buy btc

if __name__ == '__main__':
    main()