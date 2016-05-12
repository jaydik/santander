from sklearn.cross_validation import  train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


def main():

    # Load the data
    main_data = pd.read_csv('../data/main_data.csv')
    targets = pd.read_csv('../data/target.csv')

    # Split the Data
    X_train, X_test, y_train, y_test = train_test_split(main_data, targets, test_size=0.25, random_state=42)

    # Train model
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    # predict and print accuracy
    pred = rf.predict(X_test)
    print(accuracy_score(pred, y_test))

    # Get test data for submission
    test_data = pd.read_csv('../data/test_transform.csv', index_col='ID')
    test_data['TARGET'] = rf.predict(test_data)
    test_data[['TARGET']].to_csv('../submissions/randomForest.csv', index=True, index_label='ID')


if __name__ == '__main__':
    main()