import os
import time
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from brian2 import ms, mV

# Dynamic imports from previous stages
lif_mod = importlib.import_module("02_lif_simulation")
enc_mod = importlib.import_module("03_input_encoding")

build_lif_network = lif_mod.build_lif_network
encode_image = enc_mod.encode_image
get_mnist_datasets = enc_mod.get_mnist_datasets


def extract_dataset_features(
    dataset,
    net,
    G,
    spike_mon,
    n_neurons=500,
    n_input_neurons=50,
    n_samples=1000,
    sim_duration=75 * ms,
    max_current_mV=25.0,
    log_interval=100
):
    """
    Extracts spike-count feature vectors from the connectome LIF network
    for n_samples from the provided dataset.

    Returns:
        X: np.ndarray of shape (n_samples, n_neurons)
        y: np.ndarray of shape (n_samples,)
    """
    print(f"Extracting biological spike features for {n_samples} samples...")
    print(f"  Simulation duration per trial : {sim_duration}")
    print(f"  Input channels                : {n_input_neurons}")
    print(f"  Total reservoir neurons       : {n_neurons}")
    print(f"  Max injected current          : {max_current_mV} mV")

    features = []
    labels = []

    start_time = time.time()

    for idx in range(n_samples):
        img, label = dataset[idx]

        # 1. Encode image to 50 input currents
        currents = encode_image(img, n_input_neurons=n_input_neurons, max_current_mV=max_current_mV)

        # 2. Reset network state for independent trial
        prev_counts = np.array(spike_mon.count)
        G.v = -65 * mV
        G.I_ext = np.zeros(n_neurons) * mV
        G.I_ext[:n_input_neurons] = currents * mV

        # 3. Simulate trial
        net.run(sim_duration)

        # 4. Spike feature vector (count per neuron in this trial)
        counts = np.array(spike_mon.count) - prev_counts
        features.append(counts)
        labels.append(label)

        if (idx + 1) % log_interval == 0 or (idx + 1) == n_samples:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            print(f"  [{idx + 1:4d}/{n_samples}] ({rate:.1f} samples/s) - "
                  f"Last sample total spikes: {counts.sum():3d} (input: {counts[:n_input_neurons].sum():2d}, "
                  f"downstream: {counts[n_input_neurons:].sum():2d})")

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    total_time = time.time() - start_time
    print(f"\nCompleted feature extraction in {total_time:.1f}s ({total_time / n_samples * 1000:.1f} ms/sample).")
    return X, y


def run_stage4_feature_extraction(
    weight_matrix_path="outputs/W.npy",
    data_dir="./data",
    n_samples=1000,
    sim_duration=75 * ms,
    output_x_path="outputs/X_fly.npy",
    output_y_path="outputs/y_labels.npy",
    output_fig_path="outputs/figures/feature_extraction_stage4.png"
):
    print("Running Connectome Feature Extraction...")

    assert os.path.exists(weight_matrix_path), f"Connectome weight matrix not found at {weight_matrix_path}."
    W = np.load(weight_matrix_path)
    N = W.shape[0]

    # Load MNIST
    train_set, _ = get_mnist_datasets(data_dir=data_dir)

    # Build LIF network with numpy backend (no state monitor needed for batch extraction)
    print("Initializing Brian2 Connectome Network...")
    net, G, S, spike_mon, _ = build_lif_network(
        W=W,
        record_states=False,
        dt=0.5 * ms
    )

    # Warmup step
    G.I_ext = np.zeros(N) * mV
    net.run(10 * ms)

    # Extract features
    X, y = extract_dataset_features(
        dataset=train_set,
        net=net,
        G=G,
        spike_mon=spike_mon,
        n_neurons=N,
        n_input_neurons=50,
        n_samples=n_samples,
        sim_duration=sim_duration
    )

    # Save features and labels
    os.makedirs(os.path.dirname(output_x_path), exist_ok=True)
    np.save(output_x_path, X)
    np.save(output_y_path, y)
    print(f"\nSaved feature matrix to: {output_x_path} (shape: {X.shape})")
    print(f"Saved labels to: {output_y_path} (shape: {y.shape})")

    # Generate Diagnostic Visualizations
    print("\nGenerating Stage 4 Diagnostic Plots...")
    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Stage 4: Connectome Reservoir Spike Features ({n_samples} MNIST Samples)", fontsize=13, fontweight='bold')

    # Plot 1: Feature Matrix Heatmap (first 200 samples)
    n_display = min(200, n_samples)
    im = axes[0].imshow(X[:n_display, :], aspect='auto', cmap='magma', interpolation='nearest')
    axes[0].set_title(f"Spike Count Feature Matrix (First {n_display} Samples)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Neuron Index (0..49: Input, 50..499: Downstream)", fontsize=9)
    axes[0].set_ylabel("Sample Index", fontsize=9)
    axes[0].axvline(49.5, color='cyan', linestyle='--', linewidth=1, label='Input / Recurrent Boundary')
    axes[0].legend(loc='upper right', fontsize=8)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="Spike Count")

    # Plot 2: Class Mean Spike Responses
    class_means = np.zeros((10, N))
    for digit in range(10):
        class_mask = (y == digit)
        if np.any(class_mask):
            class_means[digit] = X[class_mask].mean(axis=0)

    for digit in [0, 1, 3, 7, 8]:
        axes[1].plot(class_means[digit], label=f"Digit '{digit}'", alpha=0.85, linewidth=1.2)
    axes[1].axvline(49.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].set_title("Mean Connectome Response Profile by Digit Class", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Neuron Index", fontsize=9)
    axes[1].set_ylabel("Mean Spike Count", fontsize=9)
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, linestyle=":", alpha=0.5)

    # Plot 3: 2D PCA Projection of Spike Features
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    scatter = axes[2].scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=y, cmap='tab10', s=15, alpha=0.75
    )
    axes[2].set_title(f"2D PCA of Reservoir Features (EV: {pca.explained_variance_ratio_.sum():.1%})", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("PC 1", fontsize=9)
    axes[2].set_ylabel("PC 2", fontsize=9)
    cbar = fig.colorbar(scatter, ax=axes[2], ticks=range(10), fraction=0.046, pad=0.04)
    cbar.set_label("Digit Label")
    axes[2].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_fig_path, dpi=150)
    plt.close()
    print(f"Saved diagnostic figure to: {output_fig_path}")

    assert os.path.exists(output_x_path), f"Missing {output_x_path}"
    assert os.path.exists(output_y_path), f"Missing {output_y_path}"
    assert X.shape == (n_samples, N), f"Expected shape ({n_samples}, {N}), got {X.shape}"
    assert len(y) == n_samples, f"Expected {n_samples} labels, got {len(y)}"
    assert np.count_nonzero(X) > 0, "Feature matrix contains no spikes"

    active_neurons_per_sample = (X > 0).sum(axis=1)
    mean_active = float(active_neurons_per_sample.mean())
    mean_total_spikes = float(X.sum(axis=1).mean())
    downstream_spikes_total = int(X[:, 50:].sum())

    print(f"Total Samples Processed       : {n_samples}")
    print(f"Feature Matrix Shape          : {X.shape}")
    print(f"Mean Active Neurons / Sample  : {mean_active:.1f} / {N} ({mean_active / N:.1%})")
    print(f"Mean Spikes / Sample          : {mean_total_spikes:.1f}")
    print(f"Total Downstream Spikes       : {downstream_spikes_total}")
    print(f"Variance Explained (PCA 2D)   : {pca.explained_variance_ratio_.sum():.1%}")

    assert downstream_spikes_total > 0, "No spikes reached downstream circuit"
    print("Feature extraction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Feature Extraction Loop")
    parser.add_argument("--weights", type=str, default="outputs/W.npy")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--duration", type=float, default=75.0, help="Duration in ms")
    parser.add_argument("--out-x", type=str, default="outputs/X_fly.npy")
    parser.add_argument("--out-y", type=str, default="outputs/y_labels.npy")
    parser.add_argument("--out-fig", type=str, default="outputs/figures/feature_extraction_stage4.png")
    args = parser.parse_args()

    run_stage4_feature_extraction(
        weight_matrix_path=args.weights,
        n_samples=args.samples,
        sim_duration=args.duration * ms,
        output_x_path=args.out_x,
        output_y_path=args.out_y,
        output_fig_path=args.out_fig
    )
