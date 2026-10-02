import os
import json
import time
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from brian2 import ms, mV

# Dynamic imports
lif_mod = importlib.import_module("02_lif_simulation")
enc_mod = importlib.import_module("03_input_encoding")
feat_mod = importlib.import_module("04_feature_extraction")

build_lif_network = lif_mod.build_lif_network
encode_image = enc_mod.encode_image
get_mnist_datasets = enc_mod.get_mnist_datasets
extract_dataset_features = feat_mod.extract_dataset_features


def rewire_matrix(W, seed=42):
    """
    Degree-preserving random rewiring of the weight matrix.
    Preserves:
      - Total number of synapses
      - Out-degree of every neuron
      - Exact distribution of synaptic weights
    Shuffles:
      - Target postsynaptic neuron indices
    """
    rng = np.random.default_rng(seed)
    rows, cols = np.nonzero(W)
    weights = W[rows, cols]

    perm = rng.permutation(len(cols))
    new_cols = cols[perm]

    W_random = np.zeros_like(W)
    W_random[rows, new_cols] = weights

    # Remove self-loops
    np.fill_diagonal(W_random, 0)
    return W_random


def run_stage6_random_baseline(
    w_fly_path="outputs/W.npy",
    y_path="outputs/y_labels.npy",
    fly_metrics_path="outputs/results/fly_brain_metrics.json",
    data_dir="./data",
    n_samples=1000,
    seed=42,
    test_size=0.2,
    results_dir="outputs/results",
    fig_path="outputs/figures/baseline_comparison_stage6.png"
):
    print("Running Degree-Matched Random Graph Baseline...")

    assert os.path.exists(w_fly_path), f"Fly connectome matrix not found at {w_fly_path}."
    assert os.path.exists(y_path), f"Labels not found at {y_path}."
    assert os.path.exists(fly_metrics_path), f"Fly metrics not found at {fly_metrics_path}."

    W_fly = np.load(w_fly_path)
    y = np.load(y_path)
    N = W_fly.shape[0]

    with open(fly_metrics_path, "r") as f:
        fly_metrics = json.load(f)
    acc_fly_test = fly_metrics["test_accuracy"]
    acc_fly_cv = fly_metrics["cv_mean_accuracy"]

    print(f"Loaded Biological Connectome: {N} neurons, {np.count_nonzero(W_fly)} synapses.")
    print(f"Biological Fly Brain Test Accuracy: {acc_fly_test:.3%} (5-Fold CV: {acc_fly_cv:.3%})")

    # 1. Generate Rewired Matrix
    print(f"\nGenerating degree-preserving rewired matrix (seed={seed})...")
    W_random = rewire_matrix(W_fly, seed=seed)
    w_rand_path = os.path.join("outputs", "W_random.npy")
    np.save(w_rand_path, W_random)

    print(f"Rewired Matrix Saved: {w_rand_path}")
    print(f"  Synaptic count: {np.count_nonzero(W_random)} (Fly: {np.count_nonzero(W_fly)})")
    print(f"  Weight range  : [{W_random.min():.1f}, {W_random.max():.1f}]")
    print(f"  Mean weight   : {W_random[W_random > 0].mean():.2f}")

    # 2. Extract Features from the Random Network
    train_set, _ = get_mnist_datasets(data_dir=data_dir)
    print("\nInitializing Brian2 Network with Random Rewired Matrix...")
    net_rand, G_rand, S_rand, spike_mon_rand, _ = build_lif_network(
        W=W_random,
        record_states=False,
        dt=0.5 * ms
    )
    G_rand.I_ext = np.zeros(N) * mV
    net_rand.run(10 * ms)  # warmup

    x_rand_path = os.path.join("outputs", "X_random.npy")
    X_random, _ = extract_dataset_features(
        dataset=train_set,
        net=net_rand,
        G=G_rand,
        spike_mon=spike_mon_rand,
        n_neurons=N,
        n_input_neurons=50,
        n_samples=n_samples,
        sim_duration=75 * ms
    )
    np.save(x_rand_path, X_random)
    print(f"Saved random graph feature matrix to: {x_rand_path}")

    # 3. Train and Evaluate Linear Classifier on Random Features
    print("\nTraining Linear Classifier on Random Graph Features...")
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_random, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler_r = StandardScaler()
    X_train_r_sc = scaler_r.fit_transform(X_train_r)
    X_test_r_sc = scaler_r.transform(X_test_r)

    clf_r = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    clf_r.fit(X_train_r_sc, y_train_r)

    y_pred_rand_test = clf_r.predict(X_test_r_sc)
    acc_rand_test = accuracy_score(y_test_r, y_pred_rand_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_all_r_sc = scaler_r.fit_transform(X_random)
    cv_scores_r = cross_val_score(clf_r, X_all_r_sc, y, cv=cv, scoring='accuracy')
    acc_rand_cv = cv_scores_r.mean()

    cm_random = confusion_matrix(y_test_r, y_pred_rand_test)

    # 4. Raw Input Baseline (50-D downsampled image directly to LR without reservoir)
    print("\nEvaluating Direct Raw Input Baseline (50-D sensory inputs without reservoir)...")
    X_raw = np.array([
        encode_image(train_set[i][0], n_input_neurons=50, max_current_mV=25.0)
        for i in range(n_samples)
    ])
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y, test_size=test_size, random_state=42, stratify=y
    )
    scaler_raw = StandardScaler()
    clf_raw = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    clf_raw.fit(scaler_raw.fit_transform(X_train_raw), y_train_raw)
    acc_raw_test = accuracy_score(y_test_raw, clf_raw.predict(scaler_raw.transform(X_test_raw)))

    # 5. Scientific Comparison Summary
    diff = acc_fly_test - acc_rand_test
    ratio = acc_fly_test / acc_rand_test if acc_rand_test > 0 else 0
    diff_cv = acc_fly_cv - acc_rand_cv

    print("\n" + "=" * 55)
    print("BIOLOGICAL VS RANDOM BASELINE COMPARISON")
    print("=" * 55)
    print(f"Chance Baseline Level         : 10.00%")
    print(f"Raw 50-D Input (No Reservoir) : {acc_raw_test:.3%}")
    print(f"Random Graph Test Accuracy    : {acc_rand_test:.3%} (5-Fold CV: {acc_rand_cv:.3%})")
    print(f"Fly Brain Test Accuracy       : {acc_fly_test:.3%} (5-Fold CV: {acc_fly_cv:.3%})")
    print(f"Absolute Difference (Fly - Rnd): {diff:+.3%} (CV Diff: {diff_cv:+.3%})")
    print(f"Relative Performance Ratio    : {ratio:.2f}x")
    print("=" * 55)

    # Save Results
    os.makedirs(results_dir, exist_ok=True)
    comparison_metrics = {
        "acc_chance": 0.10,
        "acc_raw_input": float(acc_raw_test),
        "acc_fly_test": float(acc_fly_test),
        "acc_fly_cv": float(acc_fly_cv),
        "acc_random_test": float(acc_rand_test),
        "acc_random_cv": float(acc_rand_cv),
        "cv_scores_random": [float(s) for s in cv_scores_r],
        "test_difference": float(diff),
        "cv_difference": float(diff_cv),
        "ratio": float(ratio)
    }

    with open(os.path.join(results_dir, "comparison_summary.json"), "w") as f:
        json.dump(comparison_metrics, f, indent=2)

    np.save(os.path.join(results_dir, "y_pred_random.npy"), y_pred_rand_test)
    np.save(os.path.join(results_dir, "cm_random.npy"), cm_random)

    # 6. Generate Diagnostic Comparison Plot
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Stage 6: Fly Brain Connectome vs Degree-Matched Random Graph Baseline", fontsize=14, fontweight='bold')

    # Panel 1: Accuracy Comparison Bar Chart
    models = ['Chance', 'Raw 50-D', 'Random Graph', 'Fly Brain']
    accs = [0.10, acc_raw_test, acc_rand_test, acc_fly_test]
    colors = ['#B0BEC5', '#78909C', '#FFA726', '#43A047']

    bars = axes[0].bar(models, accs, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].set_title('Test Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', linestyle=':', alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        axes[0].annotate(f'{h:.1%}', xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points", ha='center', fontweight='bold', fontsize=10)

    # Panel 2: Confusion Matrix Difference (Fly Brain - Random Graph)
    cm_fly = np.load(os.path.join(results_dir, "cm_fly.npy"))
    cm_fly_norm = cm_fly.astype('float') / cm_fly.sum(axis=1)[:, np.newaxis]
    cm_rand_norm = cm_random.astype('float') / cm_random.sum(axis=1)[:, np.newaxis]
    cm_diff = cm_fly_norm - cm_rand_norm

    sns.heatmap(
        cm_diff, annot=True, fmt='+.2f', cmap='RdYlGn', center=0, ax=axes[1],
        xticklabels=range(10), yticklabels=range(10), cbar=True
    )
    axes[1].set_title(r'Normalized Confusion $\Delta$ (Fly - Random)' + '\n(Green = Fly Brain Better)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Predicted Digit Class', fontsize=10)
    axes[1].set_ylabel('True Digit Class', fontsize=10)

    # Panel 3: 5-Fold Cross Validation Distributions
    cv_fly = fly_metrics["cv_scores"]
    axes[2].plot(range(1, 6), [s * 100 for s in cv_fly], marker='o', color='#43A047', linewidth=2, label=f'Fly Brain (Mean: {acc_fly_cv:.1%})')
    axes[2].plot(range(1, 6), [s * 100 for s in cv_scores_r], marker='s', color='#FFA726', linewidth=2, linestyle='--', label=f'Random Graph (Mean: {acc_rand_cv:.1%})')
    axes[2].set_xticks(range(1, 6))
    axes[2].set_xlabel('Cross-Validation Fold', fontsize=10)
    axes[2].set_ylabel('Accuracy (%)', fontsize=10)
    axes[2].set_title('5-Fold Cross Validation Trajectory', fontsize=11, fontweight='bold')
    axes[2].legend(loc='lower right', fontsize=9)
    axes[2].grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved comparison figure to: {fig_path}")

    assert os.path.exists(w_rand_path), f"Missing {w_rand_path}"
    assert os.path.exists(x_rand_path), f"Missing {x_rand_path}"
    assert os.path.exists(fig_path), f"Missing {fig_path}"
    assert W_random.shape == W_fly.shape, "Shape mismatch between random and fly graph!"
    print(f"Random baseline comparison complete (Fly: {acc_fly_test:.2%}, Random: {acc_rand_test:.2%}).")

    return comparison_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 6: Random Graph Baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()

    run_stage6_random_baseline(
        seed=args.seed,
        n_samples=args.samples
    )
