from sklearn.cross_validation import  train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import auc, roc_curve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():

    # Load the data
    print('Reading data...')
    main_data = pd.read_csv('../data/main_data.csv')
    targets = pd.read_csv('../data/target.csv')
    big_array = pd.concat([main_data, targets], axis=1)
    big_array = big_array.sample(frac=0.010)
    print(len(main_data.index))
    print(len(big_array.index))


    # Split the Data
    print('Splitting...')
    X_train, X_test, y_train, y_test = train_test_split(main_data, targets, test_size=0.25, random_state=42)

    # Train tree
    print('Training Tree...')
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    print('Predicting Tree...')
    tree_pred = tree.predict_proba(X_test)[:, 1]
    tree_fpr, tree_tpr, _ = roc_curve(y_test, tree_pred)

    # train random forest
    print('Training Random Forest...')
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    print('Predicting Random Forest...')
    rf_pred = rf.predict_proba(X_test)[:, 1]
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred)

    # train svm
    # Had to split it to a subset, way too mcuh data, too long to run
    svm_train, svm_test, svm_y_train, svm_y_test = train_test_split(big_array.drop('TARGET', axis=1),
                                                                    big_array['TARGET'],
                                                                    test_size=0.75)
    print('Training SVM...')
    svm_c = SVC(kernel='linear', probability=True)
    svm_c.fit(svm_train, svm_y_train)
    print('Predicting SVM...')
    svm_pred = svm_c.predict_proba(X_test)[:, 1]
    svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_pred)

    # plot model comparison
    print('Creating Plot...')
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(tree_fpr, tree_tpr, label='Tree')
    plt.plot(rf_fpr, rf_tpr, label='RF')
    plt.plot(svm_fpr, svm_tpr, label='SVM')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    print('Saving Plot...')
    plt.savefig('rocCurve.png')


if __name__ == '__main__':
    main()