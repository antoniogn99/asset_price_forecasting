import pandas as pd
import pickle
import config


def create_dataframes_from_prices():
    # read data
    with open(config.PRICES_FILE, 'rb') as f:
        prices_dicc = pickle.load(f)

    # create matrix
    matrix = []
    num_cols = config.INDEPENDENT_VARIABLE_DIMENSION
    for date in prices_dicc.keys():
        prices = prices_dicc[date]
        increases = [0]
        for i in range(1, len(prices)):
            increases.append(int((prices[i] - prices[i-1])*100))
        matrix += create_matrix_from_increases(increases, num_cols)

    # create list of columns names
    columns = []
    for i in range(num_cols):
        name = 'time -' + str(num_cols-i-1)
        columns.append(name)
    columns[-1] = 'time 0'
    columns.append('time +1')
    columns.append('delta')

    # create and save input dataframe
    df_input = pd.DataFrame(matrix, columns=columns)
    print(df_input)
    with open(config.INPUT_FILE, "wb") as f:
        pickle.dump(df_input, f)

    # create and save train dataframe
    train_dataframe_size = len(matrix) - config.TEST_DATAFRAME_SIZE
    df_train = pd.DataFrame(matrix[:train_dataframe_size], columns=columns)
    print(df_train)
    with open(config.TRAIN_FILE, "wb") as f:
        pickle.dump(df_train, f)

    # create and save test dataframe
    df_test = pd.DataFrame(matrix[train_dataframe_size:], columns=columns)
    print(df_test)
    with open(config.TEST_FILE, "wb") as f:
        pickle.dump(df_test, f)

def create_matrix_from_increases(increases, num_cols):
    num_rows = len(increases) - num_cols
    matrix = []
    for i in range(num_rows):
        row = increases[i:i+num_cols+1]
        if increases[i+num_cols]>0:
            row.append("POSITIVE")
        elif increases[i+num_cols]<0:
            row.append("NEGATIVE")
        else:
            row.append("ZERO")
        matrix.append(row)
    return matrix

def create_prices_diccs_per_day():
    # read data
    with open(config.PRICES_FILE, 'rb') as f:
        prices_dicc = pickle.load(f)

    dates = list(prices_dicc.keys())
    for i in range(20, len(dates)):
        aux_dic = {}
        for j in range(i-20, i+1):
            aux_dic[dates[j]] = prices_dicc[dates[j]]

        with open("../input/prices_1m_10secs/" + dates[i]+ ".pickle", "wb") as f:
            pickle.dump(aux_dic, f)


if __name__ == "__main__":
    create_dataframes_from_prices()
