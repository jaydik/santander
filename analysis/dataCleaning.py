import pandas as pd
import numpy as np
import xgboost as xgb


def main():

    main_data = pd.read_csv('../data/train.csv')

    train = xgb.DMatrix(main_data.as_matrix())
    train.save_binary('../data/xgbtrain.buffer')

if __name__ == '__main__':
    main()
