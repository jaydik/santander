import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectPercentile, f_classif


def main():

    main_data = pd.read_csv('../data/train.csv')

    output = []
    for x in main_data.columns:
        output.append({
            'variable': x,
            'variance': main_data.ix[:, x].var(),
            'corr_w_target': round(main_data.ix[:, x].corr(main_data.TARGET), 4),
            'abs_corr': abs(round(main_data.ix[:, x].corr(main_data.TARGET), 4))}
        )

    # print csv for later in the presentation docs
    variable_selector = pd.DataFrame(output)
    variable_selector = variable_selector.set_index('variable')
    variable_selector = variable_selector.drop('TARGET')
    variable_selector.sort_values('abs_corr', ascending=False).to_csv('../presentationDocs/corrs.csv')

    selector = SelectPercentile(f_classif, percentile=25)
    test = selector.fit_transform(main_data.drop('TARGET', axis=1), main_data['TARGET'])

    print(test.shape)




if __name__ == '__main__':
    main()
