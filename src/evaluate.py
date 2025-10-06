# evaluate.py
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from data_preprocess import load_data, split_and_scale

def threshold_tuning(model, X_test, y_test):
    """
    Evaluate a range of thresholds and print a selection of results.
    Returns: list of results (threshold, precision, recall, fp, fn)
    """
    probs = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.01, 0.9, 90)
    results = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        results.append((t, precision, recall, fp, fn, tp, tn))
    return results

def print_summary(results, top_n=5):
    # sort by recall desc, then by fewer false positives
    results_sorted = sorted(results, key=lambda x: (x[2], -x[3]), reverse=True)
    print(f"Top {top_n} thresholds by recall (precision, recall, fp, fn):")
    for r in results_sorted[:top_n]:
        print(f"t={r[0]:.3f}  precision={r[1]:.3f}  recall={r[2]:.3f}  fp={int(r[3])}  fn={int(r[4])}")

if __name__ == '__main__':
    # Load model
    try:
        model = joblib.load('results/models/rf_best.pkl')
    except FileNotFoundError:
        print("Model not found. Run 'python src/train_supervised.py' first.")
        raise

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    results = threshold_tuning(model, X_test, y_test)
    print_summary(results, top_n=10)

    # pick a threshold with recall >= 0.88 if exists
    candidates = [r for r in results if r[2] >= 0.88]
    if candidates:
        best = min(candidates, key=lambda x: x[3])  # least false positives among those
        print("\nChosen threshold (recall>=0.88, minimal FP):", best[0])
    else:
        # fallback to threshold with max recall
        best = max(results, key=lambda x: x[2])
        print("\nNo threshold reached recall>=0.88; best recall threshold:", best[0])
