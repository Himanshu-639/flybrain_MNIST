import os
import json
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from brian2 import ms, mV

# Dynamic imports
lif_mod = importlib.import_module("02_lif_simulation")
enc_mod = importlib.import_module("03_input_encoding")

build_lif_network = lif_mod.build_lif_network
encode_image = enc_mod.encode_image
get_mnist_datasets = enc_mod.get_mnist_datasets


def generate_final_visualizations(
    results_dir="outputs/results",
    w_path="outputs/W.npy",
    data_dir="./data",
    output_root_path="fly_brain_classifier.png",
    output_fig_path="outputs/figures/fly_brain_classifier.png",
    sample_digit_index=1  # Choose an informative sample
):
    print("Generating Final Synthesis Visualizations...")

    # 1. Load Metrics and Predictions
    comp_path = os.path.join(results_dir, "comparison_summary.json")
    cm_path = os.path.join(results_dir, "cm_fly.npy")
    y_test_path = os.path.join(results_dir, "y_test.npy")
    y_pred_path = os.path.join(results_dir, "y_pred_fly.npy")

    assert os.path.exists(comp_path), f"Missing {comp_path}. Run Stage 6 first."
    assert os.path.exists(cm_path), f"Missing {cm_path}. Run Stage 5 first."

    with open(comp_path, "r") as f:
        comp_data = json.load(f)

    acc_fly = comp_data["acc_fly_test"]
    acc_random = comp_data["acc_random_test"]
    cm = np.load(cm_path)

    print(f"Loaded Results:")
    print(f"  Fly Brain Accuracy   : {acc_fly:.3%}")
    print(f"  Random Graph Accuracy: {acc_random:.3%}")
    print(f"  Confusion Matrix     : shape {cm.shape}")

    # 2. Simulate One Informative MNIST Sample to generate Spike Raster
    print(f"\nSimulating sample #{sample_digit_index} to generate high-resolution spike raster...")
    train_set, test_set = get_mnist_datasets(data_dir=data_dir)
    sample_img, sample_label = test_set[sample_digit_index]
    print(f"Sample #{sample_digit_index} is digit '{sample_label}'.")

    W = np.load(w_path)
    N = W.shape[0]

    net, G, S, spike_mon, _ = build_lif_network(W=W, record_states=False, dt=0.2 * ms)

    # Inject encoded currents
    currents = encode_image(sample_img, n_input_neurons=50, max_current_mV=25.0)
    G.v = -65 * mV
    G.I_ext = np.zeros(N) * mV
    G.I_ext[:50] = currents * mV

    duration_ms = 75
    net.run(duration_ms * ms)

    spike_i = np.array(spike_mon.i)
    spike_t = np.array(spike_mon.t / ms)
    total_spikes = len(spike_i)
    print(f"Simulation completed: {total_spikes} total spikes across {len(np.unique(spike_i))} active neurons.")

    # Create 2D binned raster matrix (Neuron x Time ms)
    raster_matrix = np.zeros((N, duration_ms), dtype=np.float32)
    for idx, t_val in zip(spike_i, spike_t):
        t_bin = min(int(t_val), duration_ms - 1)
        raster_matrix[idx, t_bin] += 1

    # 3. Create the 3-Panel Final Figure (as defined in Stage 7 specifications)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle("Fly Brain Connectome Reservoir Computing — Final Synthesis", fontsize=14, fontweight='bold', y=1.02)

    # Panel 1: Accuracy comparison bar chart
    labels = ['Fly Brain\n(Biological)', 'Random Graph\n(Degree-Matched)']
    accs = [acc_fly, acc_random]
    colors = ['#4CAF50', '#9E9E9E']

    bars = axes[0].bar(labels, accs, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel('Test Accuracy', fontsize=11, fontweight='bold')
    axes[0].set_title('Fly Brain vs Random Graph', fontsize=12, fontweight='bold')
    axes[0].axhline(0.10, color='red', linestyle='--', linewidth=1, label='Chance (10%)')
    axes[0].grid(axis='y', linestyle=':', alpha=0.6)
    axes[0].legend(loc='lower right', fontsize=9)

    for bar, val in zip(bars, accs):
        h = bar.get_height()
        axes[0].annotate(
            f'{val:.1%}',
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5), textcoords="offset points",
            ha='center', fontweight='bold', fontsize=11
        )

    # Panel 2: Confusion matrix for fly brain
    sns.heatmap(
        cm, ax=axes[1], annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=range(10), yticklabels=range(10)
    )
    axes[1].set_title(f'Fly Brain Confusion Matrix (Acc: {acc_fly:.1%})', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Digit Class', fontsize=10)
    axes[1].set_ylabel('True Digit Class', fontsize=10)

    # Panel 3: Spike activity heatmap / raster for sample image
    # We display the 2D binned raster with 'hot' / 'magma' colormap
    im = axes[2].imshow(
        raster_matrix, aspect='auto', cmap='hot', interpolation='nearest',
        origin='lower', extent=[0, duration_ms, 0, N]
    )
    axes[2].axhline(49.5, color='cyan', linestyle='--', linewidth=1, label='Input Boundary (50)')
    axes[2].set_xlabel('Time (ms)', fontsize=10)
    axes[2].set_ylabel('Neuron Index (0..499)', fontsize=10)
    axes[2].set_title(f"Spike Raster — Sample Digit '{sample_label}'", fontsize=12, fontweight='bold')
    axes[2].legend(loc='upper right', fontsize=8)
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Spikes / ms', fontsize=9)

    plt.tight_layout()

    # Save to both paths
    plt.savefig(output_root_path, dpi=150, bbox_inches='tight')
    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
    plt.savefig(output_fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nFinal figures successfully saved to:")
    print(f"  1. {output_root_path}")
    print(f"  2. {output_fig_path}")

    assert os.path.exists(output_root_path), f"Missing {output_root_path}"
    assert os.path.exists(output_fig_path), f"Missing {output_fig_path}"
    print("Visualizations successfully generated and saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 7: Final Visualizations")
    parser.add_argument("--results-dir", type=str, default="outputs/results")
    parser.add_argument("--weights", type=str, default="outputs/W.npy")
    parser.add_argument("--out-root", type=str, default="fly_brain_classifier.png")
    parser.add_argument("--out-fig", type=str, default="outputs/figures/fly_brain_classifier.png")
    parser.add_argument("--sample-idx", type=int, default=1)
    args = parser.parse_args()

    generate_final_visualizations(
        results_dir=args.results_dir,
        w_path=args.weights,
        output_root_path=args.out_root,
        output_fig_path=args.out_fig,
        sample_digit_index=args.sample_idx
    )
