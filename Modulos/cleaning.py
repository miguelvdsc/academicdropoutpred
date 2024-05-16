    
import numpy as np
from scipy import stats
import pandas as pd
import json
from sklearn.preprocessing import OneHotEncoder

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

def translate_categorical_variables(data, columns):
    import json
    with open('traducao.json') as f:
        categorical_mapping = json.load(f)
    new_data = []
    for d in data:
        new_d = []
        for i, c in enumerate(columns):
            if c in categorical_mapping and str(d[i]) in categorical_mapping[c]:
                new_d.append(categorical_mapping[c][str(d[i])])
            elif d[i] is None:
                new_d.append(None)  # keep null values as None
            else:
                new_d.append(d[i])
        new_data.append(new_d)
    return new_data
    
def one_hot_encode(df, json_file):
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder
    import json

    # Load the JSON file
    with open(json_file) as f:
        traducao = json.load(f)

    # Identify the columns to be one-hot encoded
    columns_to_encode = list(traducao.keys())

    # Separate the non-categorical and categorical variables
    non_categorical_df = df.drop(columns_to_encode, axis=1)
    categorical_df = df[columns_to_encode]

    # Apply one-hot encoding
    enc = OneHotEncoder()
    df_encoded = pd.DataFrame(enc.fit_transform(categorical_df).toarray(), 
                              columns=enc.get_feature_names_out(input_features=columns_to_encode))
    dump(enc, 'OneHotEncoder.joblib')

    # Replace the encoded column names with the corresponding values from the JSON file
    for col in df_encoded.columns:
        feature, value = col.split('_')
        if feature in traducao and value in traducao[feature]:
            df_encoded.rename(columns={col: f'{feature}_{traducao[feature][value]}'}, inplace=True)

    # Convert the dataframes to numpy arrays and concatenate them
    df_final = pd.DataFrame(np.hstack([non_categorical_df.values, df_encoded.values]), 
                            columns=list(non_categorical_df.columns) + list(df_encoded.columns))

    return df_final

from joblib import dump, load

def oneHotEncode_pred(df, json_file):
    import pandas as pd
    import json

    # Load the JSON file
    with open(json_file) as f:
        traducao = json.load(f)

    # Load the OneHotEncoder model
    enc = load('OneHotEncoder.joblib')

    # Identify the columns to be one-hot encoded
    columns_to_encode = list(traducao.keys())
    
    
    
    # print('DataFrame columns:', df.columns)
    # print('Columns to encode:', columns_to_encode)


    # Separate the non-categorical and categorical variables
    non_categorical_df = df.drop(columns_to_encode, axis=1)
    categorical_df = df[columns_to_encode]

    # Apply the encoder to the DataFrame
    df_encoded = pd.DataFrame(enc.transform(categorical_df).toarray(), 
                              columns=enc.get_feature_names_out(input_features=columns_to_encode))

    # Replace the encoded column names with the corresponding values from the JSON file
    for col in df_encoded.columns:
        feature, value = col.split('_')
        if feature in traducao and value in traducao[feature]:
            df_encoded.rename(columns={col: f'{feature}_{traducao[feature][value]}'}, inplace=True)

    

    # Convert the dataframes to numpy arrays and concatenate them
    df_final = pd.DataFrame(np.hstack([non_categorical_df.values, df_encoded.values]), 
                            columns=list(non_categorical_df.columns) + list(df_encoded.columns))

    return df_final
    