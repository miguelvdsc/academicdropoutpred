import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from Modulos.database import get_evaluation, save_model, store_evaluation, update_model_file
from Modulos.frontend import plot_confusion_matrix


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

    # Create the confusion matrix
    cm = confusion_matrix(gtTest, gtPred)
    
    store_evaluation(id_model,cm)
    
    # Return the trained model
    
    
    #visualization
    
    plot_confusion_matrix(id_model)
    
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

        # Calculate rates
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        # Calculate total observations
        total = tp + tn + fp + fn

        # Calculate metrics
        roc_auc = metrics.auc(fpr, tpr)
        f1 = metrics.f1_score(tp, fp)
        recall = metrics.recall_score(tp, fn)
        precision = metrics.precision_score(tp, fp)
        accuracy = (tp + tn) / total

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
            "tpr": tpr,
            "fpr": fpr
        }
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}