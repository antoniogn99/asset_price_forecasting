from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.ticktype import TickTypeEnum
import datetime
import pickle
import config
import datetime
import _thread
import time
import joblib
import os

DEBUG = False
increases = []
INVERSE_STRATEGY = False

class App(EWrapper, EClient):

    price = None
    nextValidOrderId = 100
    total = 0

    def __init__(self):
        EClient.__init__(self, self)

    def error(self, reqId, errorCode, errorString):
        if DEBUG:
            print("Info: ", reqId, " ", errorCode, " ", errorString)

    def tickPrice(self, reqId, tickType, price, attrib):
        super().tickPrice(reqId, tickType, price, attrib)
        if tickType == TickTypeEnum.ASK:
            if DEBUG:
                print("TickPrice. TickerId:", reqId, "tickType:", tickType,
                      "Price:", price, "CanAutoExecute:", attrib.canAutoExecute,
                      "PastLimit:", attrib.pastLimit, "PreOpen:", attrib.preOpen)
            self.price = price

    def openOrder(self, orderId, contract, order, orderState):
        super().openOrder(orderId, contract, order, orderState)
        if DEBUG:
            print("OpenOrder. PermId: ", order.permId, "ClientId:", order.clientId, " OrderId:", orderId, 
                  "Account:", order.account, "Symbol:", contract.symbol, "SecType:", contract.secType,
                  #"Exchange:", contract.exchange, "Action:", order.action, "OrderType:", order.orderType,
                  "TotalQty:", order.totalQuantity, "CashQty:", order.cashQty, 
                  "LmtPrice:", order.lmtPrice, "AuxPrice:", order.auxPrice, "Status:", orderState.status)
        order.contract = contract
        if orderState.status == "Submitted" or orderState.status == "PreSubmitted":
            if order.orderId != 0 and order.lmtPrice != 0.0:
                print(f"{order.action} at ${order.lmtPrice}" , end="")
                if order.action == "BUY":
                    self.total -= float(order.lmtPrice)
                elif order.action == "SELL":
                    self.total += float(order.lmtPrice)
                print(f"-> Total: {self.total:.2f}")
    def orderStatus(self, orderId, status, filled,
                    remaining, avgFillPrice, permId,
                    parentId, lastFillPrice, clientId,
                    whyHeld, mktCapPrice):
        super().orderStatus(orderId, status, filled, remaining,
                            avgFillPrice, permId, parentId, lastFillPrice,
                            clientId, whyHeld, mktCapPrice)
        if DEBUG:
            print("OrderStatus. Id:", orderId, "Status:", status, "Filled:", filled,
                  "Remaining:", remaining, "AvgFillPrice:", avgFillPrice,
                  "PermId:", permId, "ParentId:", parentId, "LastFillPrice:",
                  lastFillPrice, "ClientId:", clientId, "WhyHeld:",
                  whyHeld, "MktCapPrice:", mktCapPrice)

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.nextValidOrderId = orderId

def manage_app_callbacks(app):
    app.run()

def main():
    global increases

    # create App object and connect with TWS
    app = App()
    app.connect("127.0.0.1", 7497, 0)

    # create thread to manage app callbacks
    _thread.start_new_thread( manage_app_callbacks, (app, ))

    # create conctract object
    contract = Contract()
    contract.symbol = "TSLA"
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    contract.primaryExchange = "NASDAQ"

    # request market data
    app.reqMktData(1002, contract, "", False, False, [])

    # model used to predict values
    model = joblib.load(os.path.join(config.MODEL_OUTPUT, "gbc.bin"))

    # initialize variables
    last_price = None
    last_prediction = None

    # loop until program is manually stopped
    while 1:

        print()

        # complete transaction from previous iteration
        if last_prediction == "POSITIVE":
            if INVERSE_STRATEGY:
                buy(app, contract)
            else:
                sell(app, contract)
        elif last_prediction == "NEGATIVE":
            if INVERSE_STRATEGY:
                sell(app, contract)
            else:
                buy(app, contract)

        # begin transaction from current iteration
        if app.price != None:
            if last_price == None:
                increases = [0]
                last_price = app.price
            else:
                increases.append(int(100*(app.price - last_price)))
                last_price = app.price
                if len(increases) == config.INDEPENDENT_VARIABLE_DIMENSION + 1:
                    increases = increases[1:]
                    prediction = model.predict([increases])[0]
                    print(increases, "->", prediction)
                    if prediction == "POSITIVE":
                        if INVERSE_STRATEGY:
                            sell(app, contract)
                        else:
                            buy(app, contract)
                    elif prediction == "NEGATIVE":
                        if INVERSE_STRATEGY:
                            buy(app, contract)
                        else:
                            sell(app, contract)
                    last_prediction = prediction
        if len(increases) < 10: print(increases)
        time.sleep(10)

def buy(app, contract):

    order = Order()
    order.action = "BUY"
    order.orderType = "MKT"
    order.totalQuantity = 1

    app.placeOrder(app.nextValidOrderId, contract, order)
    app.nextValidOrderId += 1

def sell(app, contract):

    order = Order()
    order.action = "SELL"
    order.orderType = "MKT"
    order.totalQuantity = 1

    app.placeOrder(app.nextValidOrderId, contract, order)
    app.nextValidOrderId += 1

if __name__ == "__main__":
    main()
