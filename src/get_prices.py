from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import datetime
import pickle
import config
import datetime
prices_dicc = {}

class App(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def error(self, reqId, errorCode, errorString):
        print("Info: ", reqId, " ", errorCode, " ", errorString)

    def contractDetails(self, reqId, contractDetails):
        print("contractDetails: ", reqId, " ", contractDetails)

    def historicalData(self, reqId, bar):
        print("HistoricalData. ReqId:", reqId, "BarData.", bar, flush=True)
        date = str(bar.date)[:8]
        if prices_dicc.get(date) == None:
            prices_dicc[date] = []
        prices_dicc[date].append(bar.close)
        with open(config.PRICES_FILE, "wb") as f:
            pickle.dump(prices_dicc, f)


def request_historical_data():
    app = App()

    app.connect("127.0.0.1", 7497, 0)

    contract = Contract()
    contract.symbol = "TSLA"
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    contract.primaryExchange = "NASDAQ"

    queryTime = datetime.datetime(2021, 7, 27).strftime("%Y%m%d %H:%M:%S")
    app.reqHistoricalData(1, contract, queryTime, "1 M", "10 secs", "TRADES", 1, 1, False, [])

    app.run()


if __name__ == "__main__":
    request_historical_data()
