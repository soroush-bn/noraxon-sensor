import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC


def train_LDA(gesture_dfs):
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

    # Step 3: Train the LDA classifier
    lda_classifier = LinearDiscriminantAnalysis()

    # Train the LDA model on the training data
    lda_classifier.fit(X_train, y_train)

    # Step 4: Evaluate the model on the test set
    y_pred = lda_classifier.predict(X_test)

    # Calculate the classification accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"LDA Accuracy: {accuracy * 100:.2f}%")

    # Print classification report (precision, recall, F1 score)
    print("LDA Report:")
    print(classification_report(y_test, y_pred))

    # Step 5: Save the trained LDA model to disk (optional)
    # joblib.dump(lda_classifier, 'lda_all_gestures_model.pkl')

    # Example of how to load the model later
    # lda_classifier = joblib.load('lda_all_gestures_model.pkl')
    # new_prediction = lda_classifier.predict(new_feature_vector)



def train_svm(gesture_dfs):
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

        # Step 3: Train the SVM classifier
        svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)  # RBF kernel is used here

        # Train the SVM model on the training data
        svm_classifier.fit(X_train, y_train)

        # Step 4: Evaluate the model on the test set
        y_pred = svm_classifier.predict(X_test)

        # Calculate the classification accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"SVM Accuracy: {accuracy * 100:.2f}%")

        # Print classification report (precision, recall, F1 score)
        print("SVM Report:")
        print(classification_report(y_test, y_pred))
