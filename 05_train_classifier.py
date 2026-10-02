import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_and_evaluate(
    x_path="outputs/X_fly.npy",
    y_path="outputs/y_labels.npy",
    results_dir="outputs/results",
    fig_path="outputs/figures/classifier_stage5.png",
    test_size=0.2,
    random_state=42
):
    print("Training Linear Readout Classifier...")

    assert os.path.exists(x_path), f"Feature matrix not found at {x_path}. Run Stage 4 first."
    assert os.path.exists(y_path), f"Labels not found at {y_path}. Run Stage 4 first."

    X = np.load(x_path)
    y = np.load(y_path)
    N_samples, N_features = X.shape

    print(f"Loaded dataset: {N_samples} samples with {N_features} features.")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # 1. Stratified Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\nDataset Split: {len(y_train)} train samples, {len(y_test)} test samples (stratified).")

    # 2. Feature Standardization
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 3. Train Logistic Regression Readout
    print("\nFitting Logistic Regression (L-BFGS, max_iter=1000)...")
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=random_state)
    clf.fit(X_train_sc, y_train)

    # 4. Predict and Evaluate on Test Set
    y_pred_train = clf.predict(X_train_sc)
    y_pred_test = clf.predict(X_test_sc)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    # 5. 5-Fold Stratified Cross Validation for robust statistics
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    X_all_sc = scaler.fit_transform(X)
    cv_scores = cross_val_score(clf, X_all_sc, y, cv=cv, scoring='accuracy')

    cm = confusion_matrix(y_test, y_pred_test)
    report = classification_report(y_test, y_pred_test, output_dict=True)

    print("\n" + "-" * 40)
    print("CLASSIFICATION PERFORMANCE SUMMARY")
    print("-" * 40)
    print(f"Fly Brain Train Accuracy : {train_acc:.3%}")
    print(f"Fly Brain Test Accuracy  : {test_acc:.3%}")
    print(f"5-Fold CV Mean Accuracy  : {cv_scores.mean():.3%} ± {cv_scores.std():.3%}")
    print(f"Baseline Chance Level    : 10.00%")
    print(f"Fold Accuracies          : {[f'{s:.3%}' for s in cv_scores]}")

    print("\nDetailed Test Classification Report:")
    print(classification_report(y_test, y_pred_test, digits=3))

    # 6. Save Evaluation Artifacts
    os.makedirs(results_dir, exist_ok=True)
    metrics = {
        "model": "Fly Brain Connectome (MaleCNS ME(R)) + Logistic Regression",
        "n_samples": int(N_samples),
        "n_features": int(N_features),
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "cv_mean_accuracy": float(cv_scores.mean()),
        "cv_std_accuracy": float(cv_scores.std()),
        "cv_scores": [float(s) for s in cv_scores],
        "classification_report": report
    }

    metrics_path = os.path.join(results_dir, "fly_brain_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(os.path.join(results_dir, "y_test.npy"), y_test)
    np.save(os.path.join(results_dir, "y_pred_fly.npy"), y_pred_test)
    np.save(os.path.join(results_dir, "cm_fly.npy"), cm)
    print(f"Saved metrics and predictions to: {results_dir}")

    # 7. Generate Stage 5 Diagnostic Plots
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Stage 5: Fly Brain Linear Classifier Performance", fontsize=13, fontweight='bold')

    # Panel 1: Normalized Confusion Matrix Heatmap
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[0],
        xticklabels=range(10), yticklabels=range(10), cbar=False
    )
    axes[0].set_title(f"Normalized Confusion Matrix (Test Acc: {test_acc:.1%})", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Predicted Digit Class", fontsize=10)
    axes[0].set_ylabel("True Digit Class", fontsize=10)

    # Panel 2: Per-Class Precision & Recall Bar Chart
    digits = [str(d) for d in range(10)]
    precisions = [report[d]['precision'] for d in digits]
    recalls = [report[d]['recall'] for d in digits]
    f1s = [report[d]['f1-score'] for d in digits]

    x_idx = np.arange(10)
    bar_w = 0.28
    axes[1].bar(x_idx - bar_w, precisions, width=bar_w, label='Precision', color='#42A5F5')
    axes[1].bar(x_idx, recalls, width=bar_w, label='Recall', color='#66BB6A')
    axes[1].bar(x_idx + bar_w, f1s, width=bar_w, label='F1-Score', color='#FFA726')
    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels(digits)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Digit Class", fontsize=10)
    axes[1].set_ylabel("Score", fontsize=10)
    axes[1].set_title("Per-Class Performance Metrics", fontsize=11, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=9)
    axes[1].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved diagnostic figure to: {fig_path}")

    assert os.path.exists(metrics_path), f"Missing {metrics_path}"
    assert os.path.exists(fig_path), f"Missing {fig_path}"
    assert test_acc > 0.10, f"Accuracy {test_acc:.2%} did not exceed chance level"
    print(f"Readout training and evaluation complete (Test Accuracy: {test_acc:.2%}).")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5: Train Linear Readout Classifier")
    parser.add_argument("--x-path", type=str, default="outputs/X_fly.npy")
    parser.add_argument("--y-path", type=str, default="outputs/y_labels.npy")
    parser.add_argument("--results-dir", type=str, default="outputs/results")
    parser.add_argument("--fig-path", type=str, default="outputs/figures/classifier_stage5.png")
    args = parser.parse_args()

    train_and_evaluate(
        x_path=args.x_path,
        y_path=args.y_path,
        results_dir=args.results_dir,
        fig_path=args.fig_path
    )
