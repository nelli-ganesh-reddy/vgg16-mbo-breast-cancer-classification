import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_and_evaluate(X_train, y_train, X_test, y_test, selected_features):
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    clf.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(clf, "models/svm_model.pkl")
    joblib.dump(selected_features, "models/selected_features.pkl")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    with open("outputs/metrics.txt", "w") as f:
        f.write(f"Test Accuracy: {acc}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred))

    print("Test Accuracy:", acc)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nModel saved in 'models/'")
    print("Results saved in 'outputs/'")