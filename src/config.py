import os


DIRECTORY = 'C:\\Users\\anton\\OneDrive\\Escritorio\\asset_price_forecasting'
INPUT_DIRECTORY = os.path.join(DIRECTORY, 'input')
PRICES_FILE = os.path.join(INPUT_DIRECTORY, 'prices.pickle')
INPUT_FILE = os.path.join(INPUT_DIRECTORY, 'df_input.pickle')
TRAIN_FILE = os.path.join(INPUT_DIRECTORY, 'df_train.pickle')
TEST_FILE = os.path.join(INPUT_DIRECTORY, 'df_test.pickle')
PRICES_DICCS_DIRECTORY = os.path.join(INPUT_DIRECTORY, 'prices_1m_10secs')
MODEL_OUTPUT = os.path.join(DIRECTORY, 'models')
NUM_FOLDS = 5
INDEPENDENT_VARIABLE_DIMENSION = 10
TEST_DATAFRAME_SIZE = 2310
