from sklearn.metrics import ConfusionMatrixDisplay, auc, precision_recall_curve, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Modulos.database import get_evaluation

def plot_confusion_matrix(id_model):
    """
    Plots the confusion matrix for a given model.

    Parameters:
    id_model (int): The ID of the model.

    Raises:
    Exception: If an error occurs while plotting the confusion matrix.

    Returns:
    None
    """
    try:
        eval = get_evaluation(id_model)
        fp = eval['fp'][0]
        fn = eval['fn'][0]
        tp = eval['tp'][0]
        tn = eval['tn'][0]

        # Plot the confusion matrix
        plt.figure()
        plt.imshow([[tp, fp], [fn, tn]], interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = ['Positive', 'Negative']
        plt.xticks([0, 1], tick_marks)
        plt.yticks([0, 1], tick_marks)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save the figure
        file = f"static/graphs/model_{id_model}_confusion_matrix.svg"
        plt.savefig(file)
        plt.close()
        

    except Exception as e:
        print(f"An error occurred: {e}")
        
def plot_roc_curve(y_test,y_score,id_modelo):
    """
    Plots the ROC curve for a given model.

    Parameters:
    y_test (numpy.ndarray): The true labels.
    y_score (numpy.ndarray): The predicted labels.
    id_model (int): The ID of the model.

    Raises:
    Exception: If an error occurs while plotting the ROC curve.

    Returns:
    None
    """
    
        # Plot the ROC curve
    plt.figure()
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
        
        # Save the figure
    file = f"static/graphs/model_{id_modelo}_roc_curve.svg"
    plt.savefig(file)
    plt.close()
        
    

def plot_precision_recall_curve(y_test,y_score,id_modelo):
    """
    Plots the precision-recall curve for a given model.

    Parameters:
    y_test (numpy.ndarray): The true labels.
    y_score (numpy.ndarray): The predicted labels.
    id_model (int): The ID of the model.

    Raises:
    Exception: If an error occurs while plotting the precision-recall curve.

    Returns:
    None
    """
    
   
        # Plot the precision-recall curve
    plt.figure()
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
        
        # Save the figure
    file = f"static/graphs/model_{id_modelo}_precision_recall_curve.svg"
    plt.savefig(file)
    plt.close()
        
    