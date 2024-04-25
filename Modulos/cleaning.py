    
import numpy as np
from scipy import stats

#metodo de imputação
def handle_nans(dataset, column, method='mean'):
    if method == 'mean':
        dataset[column] = dataset[column].fillna(dataset[column].mean())  # Replace NaNs with mean value
    elif method == 'mode':
        dataset[column] = dataset[column].fillna(dataset[column].mode().iloc[0])  # Replace NaNs with mode value
    elif method == 'median':
        dataset[column] = dataset[column].fillna(dataset[column].median())  # Replace NaNs with median value
    elif method == 'drop':
        dataset = dataset.drop(column, axis=1)  # Drop the specified column
    else:
        print('Invalid method specified. NaNs not handled.')
        
    # Perform other operations on the dataset
    
    return dataset

def handle_outliers(dataset, column, method='iqr'):
    if method == 'iqr':
        q1 = dataset[column].quantile(0.25)
        q3 = dataset[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        dataset = dataset[(dataset[column] > lower_bound) & (dataset[column] < upper_bound)]
    elif method == 'z-score':
        z = np.abs(stats.zscore(dataset[column]))
        dataset = dataset[(z < 3)]
    else:
        print('Invalid method specified. Outliers not handled.')
        
    # Perform other operations on the dataset
    
    return dataset

def handle_outliers(dataset, column, method='iqr'):
    if method == 'iqr':
        q1 = dataset[column].quantile(0.25)
        q3 = dataset[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        dataset = dataset[(dataset[column] > lower_bound) & (dataset[column] < upper_bound)]
    elif method == 'z-score':
        z = np.abs(stats.zscore(dataset[column]))
        dataset = dataset[(z < 3)]
    elif method=="categorical":
        dataset = dataset[dataset[column].isin(dataset[column].value_counts().nlargest(5).index)]
    else:
        print('Invalid method specified. Outliers not handled.')
        
    # Perform other operations on the dataset
    
    return dataset

def translate_categorical_variables(data,columns):
    import json
    with open('traducao.json') as f:
        categorical_mapping = json.load(f)
    new_data = []
    for d in data:
        new_d = []
        for i,c in enumerate(columns):
            if c in categorical_mapping:
                new_d.append(categorical_mapping[c][str(d[i])])
            else:
                new_d.append(d[i])
        new_data.append(new_d)
    return new_data
    