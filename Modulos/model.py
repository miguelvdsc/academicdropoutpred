import json
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from Modulos.database import get_evaluation, retrieve_active_model_info, retrieve_model, save_model, store_evaluation, update_model_file
from Modulos.frontend import plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve
from sklearn.preprocessing import OneHotEncoder


def train_model(dataset,params,split,id_model):
    """
    Trains a model using a decision tree algorithm.

    Parameters:
    - features: The input features for training the model.
    - labels: The corresponding labels for the input features.
    - max_depth: The maximum depth of the decision tree (default: 5).

    Returns:
    - model: The trained decision tree model.
    """
    import json
    with open('traducao.json') as f:
        categorical_mapping = json.load(f)
    # dataset = pd.read_csv("dataset 4.csv")
    transdf=translate_categorical_variabless(np.array(dataset),dataset.columns)
    
    transdf=pd.DataFrame(transdf,columns=dataset.columns)
    
    encoded_df = pd.get_dummies(transdf, columns= list(categorical_mapping.keys()))
    
    #DROP Enrolled
    
    encoded_df.drop(encoded_df[encoded_df["Target"]=='Enrolled'].index, inplace=True)
    
    
    print(encoded_df)
    print(encoded_df.columns)
    
    
    features = encoded_df.drop(columns=["Target"])
    gt=encoded_df["Target"]
    
    #model training
    
    featuresTrain,featuresTest, gtTrain, gtTest = train_test_split(features,gt,test_size=split)
    dt = tree.DecisionTreeClassifier(**params)
    dt = dt.fit(featuresTrain,gtTrain)
    
    #accuracy
    
    train_acc = dt.score(featuresTrain,gtTrain)
    print(train_acc)
    acc = dt.score(featuresTest,gtTest)
    print(acc)
    
    update_model_file(id_model,dt)
    
    
    
    #avaliação
    
    
    
    # Predict the labels for the test set
    gtPred = dt.predict(featuresTest)
    
    gtTest_numeric=gtTest.apply(lambda x: 1 if x == 'Dropout' else 0)
    gtPred_numeric=pd.Series(gtPred).apply(lambda x: 1 if x == 'Dropout' else 0)
    # Create the confusion matrix
    cm = confusion_matrix(gtTest_numeric, gtPred_numeric)
    
    print(gtTest.value_counts())    
    
    
    # Return the trained model
    tn, fp, fn, tp = cm.ravel()
    tn = tn.item() if isinstance(tn, np.int64) else tn
    fp = fp.item() if isinstance(fp, np.int64) else fp
    fn = fn.item() if isinstance(fn, np.int64) else fn
    tp = tp.item() if isinstance(tp, np.int64) else tp

    # Calculate F1 score
    f1 = f1_score(gtTest_numeric, gtPred_numeric)

    # Calculate ROC AUC
    roc_auc = roc_auc_score(gtTest_numeric, gtPred_numeric)

    # Calculate recall
    recall = recall_score(gtTest_numeric, gtPred_numeric)

    # Calculate precision
    precision = precision_score(gtTest_numeric, gtPred_numeric)

    # Calculate accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(f1,roc_auc,recall,precision,accuracy)
    
    store_evaluation(id_model,accuracy,precision,recall,roc_auc,f1,tn,fp,fn,tp)
    
    #visualization
    
    # y_score = dt.predict_proba(featuresTest)[:, 1]
    plot_confusion_matrix(id_model)
    plot_precision_recall_curve(gtTest_numeric,gtPred_numeric,id_model)
    plot_roc_curve(gtTest_numeric,gtPred_numeric,id_model)
    
    
    
    return dt
    
    
    

def translate_categorical_variabless(data,columns):
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






from sklearn import metrics

def create_full_evaluation(model_id):
    try:
        eval = get_evaluation(model_id)
        
        tn = eval['tn'].values[0]
        fp = eval['fp'].values[0]
        fn = eval['fn'].values[0]
        tp = eval['tp'].values[0]

        # Calculate F1 score
        f1 = tp / (tp + 0.5 * (fp + fn))

        # Calculate ROC AUC
        roc_auc = (tp / (tp + fn) + tn / (tn + fp)) / 2

        # Calculate recall
        recall = tp / (tp + fn)

        # Calculate precision
        precision = tp / (tp + fp)

        # Calculate accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # Return metrics, tp, tn, fp, fn, tpr, and fpr as a dictionary
        return {
            "ROC AUC": roc_auc,
            "F1": f1,
            "Recall": recall,
            "Precision": precision,
            "Accuracy": accuracy,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}
    

def predict(df, json_file):
    
    model_id = int(retrieve_active_model_info()['id_modelo'][0])
    model = retrieve_model(model_id)
    
    # Load the JSON file
    with open(json_file) as f:
        traducao = json.load(f)

    # Replace the values in the DataFrame using the JSON file
    df.replace(traducao, inplace=True)

    # Apply one-hot encoding
    enc = OneHotEncoder()
    df_encoded = pd.DataFrame(enc.fit_transform(df).toarray(), columns=enc.get_feature_names(df.columns))

    # Get the columns of the model's training data
    model_columns = model.get_booster().feature_names

    # Create a DataFrame of zeros with the same shape as the training data
    df_template = pd.DataFrame(data=np.zeros((df_encoded.shape[0], len(model_columns))), columns=model_columns)

    # Update the template with the encoded prediction data
    df_template.update(df_encoded)

    # Predict the labels for the prediction data
    predictions = model.predict(df_template)

    return predictions