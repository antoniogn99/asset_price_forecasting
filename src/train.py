import os
import config
import joblib
import pandas as pd
import pickle
import model_dispatcher
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import GradientBoostingClassifier
from numpy import mean
from numpy import absolute
from sklearn import metrics


def get_score(y, preds):
    score = 0
    for i in range(len(preds)):
        if preds[i] == "POSITIVE" and y[i] == "POSITIVE":
            score += 1
        elif preds[i] == "NEGATIVE" and y[i] == "NEGATIVE":
            score += 1
        elif preds[i] == "POSITIVE" and y[i] == "NEGATIVE":
            score -= 1
        elif preds[i] == "NEGATIVE" and y[i] == "POSITIVE":
            score -= 1

    score = 100 * score / len(preds)
    return score

def scorer(estimator, X, y):
    preds = estimator.predict(X)
    return get_score(y, preds)

def evaluate(model):
    # read the data
    with open(config.INPUT_FILE, 'rb') as f:
        df = pickle.load(f)

    # x is independent variable
    x = df.drop(["time +1", "delta"], axis=1).values

    # y is dependent variable
    y = df["delta"].values

    # define the evaluation procedure
    cv = RepeatedKFold(n_splits=config.NUM_FOLDS, n_repeats=1, random_state=1)

    # evaluate the model and collect the scores
    scores = cross_val_score(model, x, y, scoring=scorer, cv=cv, n_jobs=-1)

    return mean(scores)

def try_all_models():
    names = model_dispatcher.models.keys()
    scores = {}
    for name in names:
        print(f"Evaluating {name}")
        model = model_dispatcher.models[name]
        score = evaluate(model)
        scores[name] = score
    print(scores)

def tuning():
    # read the data
    with open(config.INPUT_FILE, 'rb') as f:
        df = pickle.load(f)

    # x is independent variable
    x = df.drop(["time +1", "delta"], axis=1).values

    # y is dependent variable
    y = df["delta"].values

    # define model
    model = GradientBoostingClassifier(random_state=1,
                                       max_depth=2,
                                       min_samples_split=1000,
                                       min_samples_leaf=10,
                                       max_features=8,
                                       subsample=0.9)

    # find best parameters
    param_test = {'learning_rate':[0.05,0.1,0.2],'n_estimators':range(30,61,10)}
    gsearch = GridSearchCV(estimator = model,
                           param_grid = param_test,
                           scoring=scorer,n_jobs=-1,
                           cv = RepeatedKFold(n_splits=config.NUM_FOLDS, n_repeats=1, random_state=1))
    gsearch.fit(x, y)
    print(gsearch.best_params_)
    print(gsearch.best_score_)

def train(model):
    # read the data
    with open(config.TRAIN_FILE, 'rb') as f:
        df = pickle.load(f)

    # x is independent variable
    x = df.drop(["time +1", "delta"], axis=1).values

    # y is dependent variable
    y = df["delta"].values

    # fit the model on the whole dataset
    model.fit(x, y)

    return model

def train_gbc():
    """
    model = GradientBoostingClassifier(random_state=1,
                                       max_depth=2,
                                       min_samples_split=1000,
                                       min_samples_leaf=10,
                                       max_features=8,
                                       subsample=0.9,
                                       n_estimators=60,
                                       learning_rate=0.1)
    """
    model = GradientBoostingClassifier(random_state=1)
    model = train(model)
    joblib.dump(model, os.path.join(config.MODEL_OUTPUT, "gbc.bin"))
    return model

def test(model):
    # read the data
    with open(config.TEST_FILE, 'rb') as f:
        df = pickle.load(f)

    # x is independent variable
    x = df.drop(["time +1", "delta"], axis=1).values

    # y is dependent variable
    y = df["delta"].values

    preds = model.predict(x)
    score = get_score(y, preds)
    print(f"Score: {score:.2f}%")


def simulation(model):
    # read the data
    with open(config.TEST_FILE, 'rb') as f:
        df = pickle.load(f)

    # x is independent variable
    x = df.drop(["time +1", "delta"], axis=1).values

    # y is dependent variable
    y = df["delta"].values

    # predicted values
    preds = model.predict(x)

    # probabilities
    probabilities = model.predict_proba(x)

    total = 0
    for index, row in df.iterrows():
            if preds[index] == "POSITIVE" and probabilities[index][1]>0.4:
                total += row["time +1"]
            elif preds[index] == "NEGATIVE" and probabilities[index][0]>0.4:
                total -= row["time +1"]

    total = total / 100
    print(f"Profit: ${total}")

    return total

def graphic_simulation(model):
    # read the data
    with open(config.TEST_FILE, 'rb') as f:
        df = pickle.load(f)

    # x is independent variable
    x = df.drop(["time +1", "delta"], axis=1).values

    # y is dependent variable
    y = df["delta"].values

    # predicted values
    preds = model.predict(x)

    # calculate profit
    total = 0
    for index, row in df.iterrows():
        if preds[index] == "POSITIVE":
            total += row["time +1"]
        elif preds[index] == "NEGATIVE":
            total -= row["time +1"]
    total = total / 100
    print(f"Profit: ${total}")

    # graphics
    import matplotlib.pyplot as plt
    import numpy as np
    import time
    max_len = 40
    margin = 10

    # get values
    values = df.values

    # show figure
    plt.xlim([0, max_len+margin])
    plt.ion()
    plt.show()

    # initalize list of prices
    price = 627.15
    prices = [price]
    increases = values[0][:config.INDEPENDENT_VARIABLE_DIMENSION]
    for i in range(len(increases)):
        price = price + increases[i]/100
        prices.append(price)

    # initialize list of types
    types = [None]*len(prices)

    # plot prices
    x = range(len(prices))
    plt.plot(x, prices, marker=".", markersize=1, color='black')
    plt.draw()

    for i in range(len(values)):
        # get the corresponding row
        row = values[i]

        # reset ficgure
        plt.clf()
        plt.xlim([0, max_len+margin])

        # update list of prices
        price = price + row[config.INDEPENDENT_VARIABLE_DIMENSION]/100
        prices.append(price)
        if len(prices) == max_len + 1:
            prices = prices[1:]
        x = range(len(prices))

        # plot prices
        plt.plot(x, prices, marker=".", markersize=1, color='black')

        # update list of types
        if preds[i] == "POSITIVE" and row[config.INDEPENDENT_VARIABLE_DIMENSION]>0:
            types.append("WINNING")
        elif preds[i] == "NEGATIVE" and row[config.INDEPENDENT_VARIABLE_DIMENSION]<0:
            types.append("WINNING")
        elif preds[i] == "POSITIVE" and row[config.INDEPENDENT_VARIABLE_DIMENSION]<0:
            types.append("LOSING")
        elif preds[i] == "NEGATIVE" and row[config.INDEPENDENT_VARIABLE_DIMENSION]>0:
            types.append("LOSING")
        else:
            types.append(None)
        if len(types) == max_len + 1:
            types = types[1:]

        # plot if there are points with special types
        for j in range(1,len(types)):
            if types[j] == "WINNING":
                plt.plot([j-1,j], [prices[j-1], prices[j]], marker=".", markersize=1, color='green')
            if types[j] == "LOSING":
                plt.plot([j-1,j], [prices[j-1], prices[j]], marker=".", markersize=1, color='red')
                

        plt.draw()
        plt.pause(0.1)

    plt.pause(100)

def exploring():
    import dataframe_creator
    import os
    from statistics import mean

    filenames = os.listdir(config.PRICES_DICCS_DIRECTORY)
    filenames.sort()
    #filenames = filenames[:2]
    dic = {}
    for filename in filenames:
        config.PRICES_FILE = config.PRICES_DICCS_DIRECTORY + filename
        dataframe_creator.create_dataframes_from_prices()
        model = train_gbc()
        test(model)
        profits = simulation(model)
        date = filename[:8]
        dic[date] = profits

    print("\nProfits per day")
    for date in dic.keys():
        print(f"{date[:4]}-{date[4:6]}-{date[6:]}: ${dic[date]}")

    m = mean(dic.values())
    print(f"\nMean: ${m:.2f}")

    s = sum(dic.values())
    print(f"\nTotal: ${s:.2f}")

    print("\n\n\n\n")
    print(dic)

if __name__ == "__main__":
    config.TRAIN_FILE = config.INPUT_FILE
    train_gbc()
    print("Model ready")
