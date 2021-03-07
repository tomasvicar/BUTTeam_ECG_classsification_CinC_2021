from test_model import test_model
from config import Config






if __name__ == '__main__':

    

    # resave_valid_data(Config.DATA,Config.DATA_RESAVE_PATH + '/test_data')

    model_directory = Config.DATA_RESAVE_PATH
    data_directory = Config.
    output_directory = Config

    test_model(model_directory, data_directory, output_directory)