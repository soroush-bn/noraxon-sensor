import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from feature_extraction import sffs
import optuna
import yaml
import joblib
import os 
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
directory = os.path.join(config["saving_dir"], f"{config['first_name']}_{config['last_name']}_{config['experiment_name']}")

def train_LDA(gesture_dfs):
    # if config["load_model"]==True : 

    # Step 1: Combine all gesture DataFrames into one large DataFrame
    all_data = pd.DataFrame()  # Combined DataFrame for all gestures
    all_labels = []  # Store labels for each row in all_data

    for i, gesture_df in enumerate(gesture_dfs):
        # Combine each gesture's DataFrame into one large DataFrame
        all_data = pd.concat([all_data, gesture_df], axis=0)
        all_labels.extend([i] * len(gesture_df))  # Create labels corresponding to the gesture

    # Convert labels to a NumPy array
    all_labels = np.array(all_labels)

    # Step 2: Split the data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

    if config["feature_selection"]==True:
        selected_features = sffs(X_train, y_train, max_features=10)
        print(f"Selected Features: {selected_features}")
        X_train = X_train[:, selected_features]
        X_test = X_test[:, selected_features]

    # Step 3: Define the objective function for Optuna
    def objective(trial):
        """Objective function for Bayesian Optimization."""
        # Suggest the parameters for 'shrinkage' and 'tol'
        shrinkage = trial.suggest_uniform('shrinkage', 0.0, 1.0)  # Continuous value in [0, 1]
        tol = trial.suggest_loguniform('tol', 1e-6, 1e-1)  # Log-uniform for tolerance (1e-6 to 1e-1)
        
        # Train the LDA model
        lda_classifier = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=shrinkage, tol=tol)
        
        # Train using cross-validation
        lda_classifier.fit(X_train, y_train)
        
        # Evaluate on the test set
        y_pred = lda_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy  # We want to maximize the accuracy

    # Step 4: Run the Bayesian Optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)  # Run for 50 trials (can be increased for better results)
    
    # Get the best parameters and the best score
    best_params = study.best_params
    best_score = study.best_value

    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-Validated Accuracy: {best_score * 100:.2f}%")

    # Step 5: Train the final LDA classifier using the best parameters
    final_lda_classifier = LinearDiscriminantAnalysis(
        solver='lsqr',
        shrinkage=best_params['shrinkage'],
        tol=best_params['tol']
    )
    final_lda_classifier.fit(X_train, y_train)

    # Step 6: Evaluate the final model on the test set
    y_pred = final_lda_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Final LDA Accuracy: {accuracy * 100:.2f}%")

    # Print the classification report
    print("LDA Classification Report:")
    print(classification_report(y_test, y_pred))
    joblib.dump(final_lda_classifier, os.path.join(directory,'final_lda_model.pkl'))

    # Example of how to load the model later
    # lda_classifier = joblib.load('lda_all_gestures_model.pkl')
    # new_prediction = lda_classifier.predict(new_feature_vector)


def train_svm(gesture_dfs):
    # Step 1: Combine all gesture DataFrames into one large DataFrame
    all_data = pd.DataFrame()  # Combined DataFrame for all gestures
    all_labels = []  # Store labels for each row in all_data

    for i, gesture_df in enumerate(gesture_dfs):
        # Combine each gesture's DataFrame into one large DataFrame
        all_data = pd.concat([all_data, gesture_df], axis=0)
        all_labels.extend([i] * len(gesture_df))  # Create labels corresponding to the gesture

    # Convert labels to a NumPy array
    all_labels = np.array(all_labels)

    # Step 2: Split the data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

    if config["feature_selection"]==True:
        selected_features = sffs(X_train, y_train, max_features=10)
        print(f"Selected Features: {selected_features}")
        X_train = X_train[:, selected_features]
        X_test = X_test[:, selected_features]

    # Step 3: Define the objective function for Optuna
    def objective(trial):
        """Objective function for Bayesian Optimization."""
        # Suggest the parameters for 'C' and 'kernel'
        C = trial.suggest_loguniform('C', 1e-3, 1e3)  # Log-uniform for C (range: 0.001 to 1000)
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        
        # Train the SVM model
        svm_classifier = SVC(C=C, kernel=kernel, probability=True, random_state=42)
        
        # Train using cross-validation
        svm_classifier.fit(X_train, y_train)
        
        # Evaluate on the test set
        y_pred = svm_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy  # We want to maximize the accuracy

    # Step 4: Run the Bayesian Optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)  # Run for 50 trials (can be increased for better results)
    
    # Get the best parameters and the best score
    best_params = study.best_params
    best_score = study.best_value

    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-Validated Accuracy: {best_score * 100:.2f}%")

    # Step 5: Train the final SVM classifier using the best parameters
    final_svm_classifier = SVC(
        C=best_params['C'],
        kernel=best_params['kernel'],
        probability=True,
        random_state=42
    )
    final_svm_classifier.fit(X_train, y_train)

    # Step 6: Evaluate the final model on the test set
    y_pred = final_svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Final SVM Accuracy: {accuracy * 100:.2f}%")

    # Print the classification report
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(final_svm_classifier, os.path.join(directory,'final_svm_model.pkl'))
