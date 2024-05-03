from sklearn.metrics import ConfusionMatrixDisplay
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