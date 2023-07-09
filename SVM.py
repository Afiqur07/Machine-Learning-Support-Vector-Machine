#
#  Assignment 3
#
#  Group 33:
#  <SM Afiqur Rahman> <smarahman@mun.ca>
#  <Jubaer Ahmed Bhuiyan> <jabhuiyan@mun.ca>

####################################################################################
# Imports
####################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, balanced_accuracy_score, accuracy_score, confusion_matrix, classification_report
import tkinter as tk

#####################################################################################
#  Implementation - helper functions and question wise result generators
#####################################################################################


def classify():
    print('Performing classification...')


def Q1_results():
    print('Generating results for Q1...')
    # Load the training data
    train_sNC = pd.read_csv("train.fdg_pet.sNC.csv", header=None)
    train_sDAT = pd.read_csv("train.fdg_pet.sDAT.csv", header=None)

    # Concatenate the training data and create labels
    X_train = pd.concat([train_sNC, train_sDAT])
    y_train = np.concatenate(
        [np.zeros(len(train_sNC)), np.ones(len(train_sDAT))])

    # Load the test data
    test_sNC = pd.read_csv("test.fdg_pet.sNC.csv", header=None)
    test_sDAT = pd.read_csv("test.fdg_pet.sDAT.csv", header=None)

    # Concatenate the test data and create labels
    X_test = pd.concat([test_sNC, test_sDAT])
    y_test = np.concatenate([np.zeros(len(test_sNC)), np.ones(len(test_sDAT))])

    # Split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    # Define the parameter grid to search over
    param_grid = {'C': [0.1, 1, 10, 100, 1000]}

    # Create a SVM classifier object
    clf = SVC(kernel='linear')

    # Create a GridSearchCV object and fit it to the training data
    grid_search = GridSearchCV(clf, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameter setting
    print("Best hyperparameter setting: ", grid_search.best_params_)

    # Plot the performance of the models explored during the C hyperparameter tuning phase
    C_values = [0.1, 1, 10, 100, 1000]
    mean_scores = grid_search.cv_results_['mean_test_score']
    std_scores = grid_search.cv_results_['std_test_score']
    plt.plot(C_values, mean_scores, marker='o')
    plt.fill_between(C_values, mean_scores - std_scores,
                     mean_scores + std_scores, alpha=0.2)
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('Mean cross-validated accuracy')
    plt.title('Performance of linear SVM as a function of C')
    plt.show()

    # Create and train the final linear SVM classifier with the best hyperparameter setting
    clf = SVC(kernel='linear', C=grid_search.best_params_['C'])
    clf.fit(X_train, y_train)

    # Evaluate the final model on the test data
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Calculate the sensitivity, specificity, precision, recall, and balanced accuracy
    tn, fp, fn, tp = confusion.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    recall = sensitivity
    balanced_accuracy = (sensitivity + specificity) / 2

    # Print the performance metrics of the final model
    print("Accuracy:", accuracy)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Balanced Accuracy:", balanced_accuracy)
    print("Confusion matrix:\n", confusion)
    print("Classification report:\n", report)

    # Create a new window to display the performance metrics
    root = tk.Tk()
    root.geometry("500x300")
    root.title("Performance Metrics")

    # Create a label to display the performance metrics
    metrics_label = tk.Label(root, text="Accuracy: {}\nSensitivity: {}\nSpecificity: {}\nPrecision: {}\nRecall: {}\nBalanced Accuracy: {}\nConfusion Matrix:\n{}".format(
        accuracy, sensitivity, specificity, precision, recall, balanced_accuracy, confusion))
    metrics_label.pack()

    # Show the window
    root.mainloop()


def Q2_results():
    print('Generating results for Q2...')

    # Load training data
    train_sNC = pd.read_csv("train.fdg_pet.sNC.csv", header=None)
    train_sDAT = pd.read_csv("train.fdg_pet.sDAT.csv", header=None)

    train_sNC["label"] = 0
    train_sDAT["label"] = 1
    train_data = pd.concat([train_sNC, train_sDAT])

    X_train = train_data.drop("label", axis=1)
    y_train = train_data["label"]

    # Load test data
    test_sNC = pd.read_csv("test.fdg_pet.sNC.csv", header=None)
    test_sDAT = pd.read_csv("test.fdg_pet.sDAT.csv", header=None)

    test_sNC["label"] = 0
    test_sDAT["label"] = 1
    test_data = pd.concat([test_sNC, test_sDAT])

    X_test = test_data.drop("label", axis=1)
    y_test = test_data["label"]

    # Perform hyperparameter tuning using GridSearchCV
    # List of values to explore for regularization parameter C and degree of polynomial kernel
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 15], 'degree': [2, 3]}
    svm = SVC(kernel='poly')  # Polynomial kernel SVM model
    # 5-fold cross-validation
    grid_search = GridSearchCV(svm, param_grid, cv=5)
    grid_search.fit(X_train, y_train)  # Fit the model

    # Extract the best hyperparameter setting
    best_C = grid_search.best_params_['C']
    best_degree = grid_search.best_params_['degree']

    # Train the final model using the best hyperparameter setting on the entire training dataset
    svm_final = SVC(kernel='poly', C=best_C, degree=best_degree)
    svm_final.fit(X_train, y_train)

    # Predict on the test dataset
    y_pred = svm_final.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    specificity = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    # Print performance metrics
    print("Accuracy: {:.4f}".format(accuracy))
    print("Sensitivity: {:.4f}".format(sensitivity))
    print("Specificity: {:.4f}".format(specificity))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("Balanced Accuracy: {:.4f}".format(balanced_accuracy))


def Q3_results():
    print('Generating results for Q3...')

    # Load training data
    train_sNC = pd.read_csv("train.fdg_pet.sNC.csv", header=None)
    train_sDAT = pd.read_csv("train.fdg_pet.sDAT.csv", header=None)

    train_sNC["label"] = 0
    train_sDAT["label"] = 1
    train_data = pd.concat([train_sNC, train_sDAT])

    X_train = train_data.drop("label", axis=1)
    y_train = train_data["label"]

    # Load test data
    test_sNC = pd.read_csv("test.fdg_pet.sNC.csv", header=None)
    test_sDAT = pd.read_csv("test.fdg_pet.sDAT.csv", header=None)

    test_sNC["label"] = 0
    test_sDAT["label"] = 1
    test_data = pd.concat([test_sNC, test_sDAT])

    X_test = test_data.drop("label", axis=1)
    y_test = test_data["label"]

    # Perform hyperparameter tuning using GridSearchCV
    param_grid = {'C': np.logspace(-3, 3, 7), 'gamma': np.logspace(-3, 3, 7)}
    svm = SVC(kernel='rbf')
    grid_search = GridSearchCV(svm, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameter setting
    best_C = grid_search.best_params_['C']
    best_gamma = grid_search.best_params_['gamma']

    # Train the final RBF kernel SVM model on the entire training dataset with the best hyperparameter setting
    svm_final = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
    svm_final.fit(X_train, y_train)

    # Predict on the test dataset
    y_pred = svm_final.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    # Print performance metrics
    print("Accuracy: {:.4f}".format(accuracy))
    print("Sensitivity: {:.4f}".format(sensitivity))
    print("Specificity: {:.4f}".format(specificity))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("Balanced Accuracy: {:.4f}".format(balanced_accuracy))


def diagnoseDAT(Xtest, data_dir):
    """Returns a vector of predictions with elements "0" for sNC and "1" for sDAT,
    corresponding to each of the N_test features vectors in Xtest
    Xtest: N_test x 14 matrix of test feature vectors
    data_dir: full path to the folder containing the following files:
    train.fdg_pet.sNC.csv, train.fdg_pet.sDAT.csv,
    test.fdg_pet.sNC.csv, test.fdg_pet.sDAT.csv
    """

    # Load the required datasets
    train_NC = pd.read_csv(data_dir + "/train.fdg_pet.sNC.csv", header=None)
    train_DAT = pd.read_csv(data_dir + "/train.fdg_pet.sDAT.csv", header=None)

    # Combine the two training datasets and add labels
    train_NC["label"] = 0
    train_DAT["label"] = 1
    train_data = pd.concat([train_NC, train_DAT])

    # Split the training data into features (X) and labels (y)
    X_train = train_data.drop("label", axis=1)
    y_train = train_data["label"]

    # Train the "best" SVM model using C =1, degree = 3
    svm = SVC(kernel='poly', C=1, degree=3)
    svm.fit(X_train, y_train)

    return svm.predict(Xtest)


#########################################################################################
# Calls to generate the results
#########################################################################################
if __name__ == "__main__":
    Q1_results()
    Q2_results()
    Q3_results()
