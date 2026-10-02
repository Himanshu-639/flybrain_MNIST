import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def get_mnist_datasets(data_dir="./data"):
    """
    Downloads and returns MNIST train and test datasets.
    """
    os.makedirs(data_dir, exist_ok=True)
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return train_set, test_set


def encode_image(image_tensor, n_input_neurons=50, max_current_mV=25.0):
    """
    Flattens and downsamples a 28x28 (784 pixel) MNIST image to n_input_neurons,
    scaling values to an external current range in mV.

    Args:
        image_tensor: torch.Tensor of shape (1, 28, 28) or (28, 28) with values in [0, 1].
        n_input_neurons: Number of input neurons to stimulate (default: 50).
        max_current_mV: Maximum external injected current (in mV).
                        Since v_thresh - v_rest = 15 mV, max_current > 15 mV ensures
                        bright pixels drive action potentials.

    Returns:
        np.ndarray of shape (n_input_neurons,) with values in [0, max_current_mV].
    """
    if isinstance(image_tensor, torch.Tensor):
        flat = image_tensor.detach().cpu().numpy().flatten()
    elif isinstance(image_tensor, np.ndarray):
        flat = image_tensor.flatten()
    else:
        raise TypeError(f"Unsupported image type: {type(image_tensor)}")

    total_pixels = len(flat)
    chunk_size = total_pixels // n_input_neurons

    # Contrast-adaptive pooling across pixel slices (photoreceptor adaptation)
    compressed = np.zeros(n_input_neurons, dtype=np.float32)
    for i in range(n_input_neurons):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < n_input_neurons - 1 else total_pixels
        slice_pixels = flat[start_idx:end_idx]
        # Mix of average intensity and peak stroke edge
        compressed[i] = 0.5 * slice_pixels.mean() + 0.5 * slice_pixels.max()

    # Contrast normalization across the visual field
    peak = compressed.max()
    if peak > 1e-6:
        compressed = compressed / peak

    # Scale to voltage drive [0, max_current_mV]
    currents = compressed * max_current_mV
    return currents


def run_stage3_verification(
    data_dir="./data",
    n_input_neurons=50,
    max_current_mV=25.0,
    output_fig_path="outputs/figures/input_encoding_stage3.png"
):
    print("Loading MNIST dataset and verifying input encoding...")

    print(f"Loading MNIST dataset from {data_dir}...")
    train_set, test_set = get_mnist_datasets(data_dir=data_dir)
    print(f"Loaded MNIST: {len(train_set)} training samples, {len(test_set)} test samples.")

    # Find one representative image for each digit 0..9
    digit_samples = {}
    for img, label in train_set:
        if label not in digit_samples:
            digit_samples[label] = img
        if len(digit_samples) == 10:
            break

    print(f"\nEncoding sample images for digits 0–9 into {n_input_neurons} input currents (max {max_current_mV} mV)...")
    encoded_profiles = {}
    for digit in range(10):
        img = digit_samples[digit]
        currents = encode_image(img, n_input_neurons=n_input_neurons, max_current_mV=max_current_mV)
        encoded_profiles[digit] = currents
        suprathreshold_count = np.sum(currents >= 15.0)  # currents >= 15 mV trigger spikes
        print(f"  Digit {digit}: min={currents.min():.2f} mV, max={currents.max():.2f} mV, "
              f"mean={currents.mean():.2f} mV, suprathreshold bins={suprathreshold_count}/{n_input_neurons}")

    # Check pairwise cosine similarities between encoded digits to verify distinctiveness
    feature_matrix = np.array([encoded_profiles[d] for d in range(10)])  # (10, 50)
    norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True) + 1e-8
    normalized = feature_matrix / norms
    similarity_matrix = normalized @ normalized.T

    # Generate Visualization
    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 10, height_ratios=[1.2, 1.2, 1.4])

    fig.suptitle("Stage 3: MNIST Input Encoding Pipeline (28×28 Pixels → 50 Input Currents)", fontsize=14, fontweight='bold')

    # Row 1: Original Images (0–9)
    for digit in range(10):
        ax_img = fig.add_subplot(gs[0, digit])
        img_np = digit_samples[digit].squeeze().numpy()
        ax_img.imshow(img_np, cmap='gray_r')
        ax_img.set_title(f"Digit '{digit}'", fontsize=11, fontweight='bold')
        ax_img.axis('off')

    # Row 2: Encoded 50-neuron Current Profiles
    for digit in range(10):
        ax_bar = fig.add_subplot(gs[1, digit])
        currents = encoded_profiles[digit]
        ax_bar.bar(range(n_input_neurons), currents, color='#1E88E5', width=1.0)
        ax_bar.axhline(15.0, color='red', linestyle='--', linewidth=0.8, alpha=0.8)
        ax_bar.set_ylim(0, max_current_mV + 2)
        if digit == 0:
            ax_bar.set_ylabel("Injected Current (mV)", fontsize=9)
        else:
            ax_bar.set_yticklabels([])
        ax_bar.set_xticks([0, 25, 49])
        ax_bar.tick_params(labelsize=8)

    # Row 3: Cross-digit Correlation & Average Profile
    ax_sim = fig.add_subplot(gs[2, :5])
    cax = ax_sim.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
    fig.colorbar(cax, ax=ax_sim, fraction=0.046, pad=0.04, label="Cosine Similarity")
    ax_sim.set_xticks(range(10))
    ax_sim.set_yticks(range(10))
    ax_sim.set_xticklabels(range(10))
    ax_sim.set_yticklabels(range(10))
    ax_sim.set_title("Cross-Digit Cosine Similarity Matrix", fontsize=11, fontweight='bold')
    ax_sim.set_xlabel("Digit Class")
    ax_sim.set_ylabel("Digit Class")

    ax_dist = fig.add_subplot(gs[2, 5:])
    for d in [0, 1, 4, 8]:
        ax_dist.plot(encoded_profiles[d], label=f"Digit {d}", linewidth=1.5, alpha=0.85)
    ax_dist.axhline(15.0, color='red', linestyle='--', linewidth=1, label=r"Threshold Drive ($I_{ext} \geq 15$ mV)")
    ax_dist.set_xlabel("Input Neuron Index (0..49)", fontsize=10)
    ax_dist.set_ylabel("Injected Current (mV)", fontsize=10)
    ax_dist.set_title("Input Current Profiles across Contrasting Digits (0, 1, 4, 8)", fontsize=11, fontweight='bold')
    ax_dist.legend(loc="upper right", fontsize=9)
    ax_dist.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_fig_path, dpi=150)
    plt.close()
    print(f"\nSaved Stage 3 diagnostic figure to: {output_fig_path}")

    for digit in range(10):
        currents = encoded_profiles[digit]
        assert currents.shape == (n_input_neurons,), f"Expected shape ({n_input_neurons},), got {currents.shape}"
        assert currents.min() >= 0.0, f"Negative current detected for digit {digit}"
        assert currents.max() <= max_current_mV + 1e-4, f"Current exceeded max for digit {digit}"
        assert np.sum(currents >= 15.0) > 0, f"No suprathreshold inputs generated for digit {digit}"

    assert os.path.exists(output_fig_path), f"Figure {output_fig_path} not found"
    print("Input encoding successfully validated across all digit classes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Input Encoding Verification")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory for MNIST data")
    parser.add_argument("--n-inputs", type=int, default=50, help="Number of input neurons")
    parser.add_argument("--max-current", type=float, default=25.0, help="Maximum injected current in mV")
    parser.add_argument("--fig-path", type=str, default="outputs/figures/input_encoding_stage3.png")
    args = parser.parse_args()

    run_stage3_verification(
        data_dir=args.data_dir,
        n_input_neurons=args.n_inputs,
        max_current_mV=args.max_current,
        output_fig_path=args.fig_path
    )
