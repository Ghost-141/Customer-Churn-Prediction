import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(model_path, X_test, y_test):
    """
    Function to load a model from a pickle file, make predictions on the test set,
    and plot the confusion matrix as a heatmap.

    Parameters:
    - model_path: str, path to the saved model pickle file
    - X_test: pandas DataFrame or numpy array, features of the test set
    - y_test: pandas Series or numpy array, actual target values for the test set
    """
    # Load the trained model from pickle file

    model = joblib.load(model_path)
    
    # Predict using the loaded model
    y_test_pred = model.predict(X_test)
    
    # Generate the confusion matrix
    cf_knn = confusion_matrix(y_test, y_test_pred)
    
    # Plot the confusion matrix using a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cf_knn, annot=True, cmap="flare", linecolor="orange", fmt="d", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model_performance(path, X_test, y_test):
    """
    Function to evaluate model performance with Accuracy, Precision, Recall, F1-score, and ROC-AUC.

    Parameters:
    - model: The trained classification model
    - X_test: Test set features
    - y_test: True target labels for the test set
    """
    
    model = joblib.load(path)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for ROC-AUC

    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate Precision, Recall, and F1-score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Calculate ROC-AUC score
    roc_auc = roc_auc_score(y_test, y_prob)

    # Print the metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Plot the ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
