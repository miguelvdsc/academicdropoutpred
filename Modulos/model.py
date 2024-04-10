from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def train_model(features, labels, max_depth=5):
    """
    Trains a model using a decision tree algorithm.

    Parameters:
    - features: The input features for training the model.
    - labels: The corresponding labels for the input features.
    - max_depth: The maximum depth of the decision tree (default: 5).

    Returns:
    - model: The trained decision tree model.
    """

    # Split the data into training and evaluation sets
    X_train, X_eval, y_train, y_eval = train_test_split(features, labels, test_size=0.3, random_state=42)

    # Create a decision tree classifier
    model = DecisionTreeClassifier(max_depth=max_depth, criterion='gini', splitter='best', min_samples_split=2,
                                   min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
                                   max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                   class_weight=None, presort=False)

    # Train the model using the training data
    model.fit(X_train, y_train)

    # Evaluate the model using the evaluation data
    accuracy = model.score(X_eval, y_eval)
    print(f"Model accuracy: {accuracy}")

    # Return the trained model
    return model