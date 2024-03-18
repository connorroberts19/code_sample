import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, RocCurveDisplay)
import seaborn as sns

class ModelMetrics:
    def __init__(self, y_true, y_pred, y_score):
        '''
        Initialize the class with true labels, predicted labels, and score probabilities.
        '''
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_score = y_score

    def report_metrics(self):
        '''Calculates and prints the desired metrics.'''
        print("Accuracy:", accuracy_score(self.y_true, self.y_pred))
        print("Precision:", precision_score(self.y_true, self.y_pred, average='binary'))
        print("Recall:", recall_score(self.y_true, self.y_pred, average='binary'))
        print("F1 Score:", f1_score(self.y_true, self.y_pred, average='binary'))
        print("ROC AUC Score:", roc_auc_score(self.y_true, self.y_score))

    def plot_confusion_matrix(self):
        '''Plots the confusion matrix'''
        cm = confusion_matrix(self.y_true, self.y_pred)
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalise
        sns.heatmap(cmn, annot=True, fmt='.2f')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show(block=False)

    def plot_roc_curve(self):
        '''Plots the ROC curve'''
        RocCurveDisplay.from_predictions(self.y_true, self.y_score)
        plt.title('ROC Curve')
        plt.show()
