# train_supervised.py
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, roc_curve

from data_preprocess import load_data, split_and_scale

# ensure output directories
os.makedirs('results/models', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

def train_rf(X_train, y_train, params=None):
    """
    Train RandomForest with SMOTE applied to the training data.
    params: dict for RF hyperparameters (optional)
    """
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    if params is None:
        params = {
            'n_estimators': 200,
            'max_depth': 110,
            'max_features': 4,
            'min_samples_split': 5,
            'random_state': 42,
            'n_jobs': -1
        }
    rf = RandomForestClassifier(**params)
    rf.fit(X_res, y_res)
    return rf

def evaluate_model(model, X_test, y_test, threshold=0.5, save_model=True):
    """
    Evaluate model on test set, save ROC and PRC plots, and optionally save model to results/models/.
    """
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    print("Classification report (threshold = {:.2f}):".format(threshold))
    print(classification_report(y_test, preds, digits=4))

    auc = roc_auc_score(y_test, probs)
    print("ROC AUC:", auc)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC={auc:.4f}')
    plt.plot([0,1],[0,1],'--', alpha=0.5)
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/figures/roc.png')
    plt.close()

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, probs)
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision)
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve')
    plt.tight_layout()
    plt.savefig('results/figures/prc.png')
    plt.close()

    if save_model:
        joblib.dump(model, 'results/models/rf_best.pkl')
        print("Saved model to results/models/rf_best.pkl")

if __name__ == '__main__':
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    print("Training RandomForest (with SMOTE)...")
    rf = train_rf(X_train, y_train)
    print("Evaluating with threshold 0.5")
    evaluate_model(rf, X_test, y_test, threshold=0.5)
    print("You can later run evaluate.py to tune thresholds.")
