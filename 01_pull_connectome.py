import os
import sys
import argparse
import numpy as np
import pandas as pd
from neuprint import Client, fetch_neurons, fetch_adjacencies
from neuprint import NeuronCriteria as NC

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Pull visual subcircuit from neuPrint MaleCNS")
    parser.add_argument("--token", type=str, default=None, help="neuPrint auth token")
    parser.add_argument("--dataset", type=str, default="male-cns:v1.0", help="neuPrint dataset name")
    parser.add_argument("--server", type=str, default="neuprint.janelia.org", help="neuPrint server host")
    parser.add_argument("--roi", type=str, default="ME(R)", help="Target ROI (default: Right Medulla ME(R))")
    parser.add_argument("--min-pre", type=int, default=5, help="Minimum presynaptic count")
    parser.add_argument("--max-neurons", type=int, default=500, help="Maximum number of neurons to fetch")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    # Determine token from arg, env variable, or .env file
    token = args.token or os.environ.get("NEUPRINT_TOKEN") or os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    if not token and os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("NEUPRINT_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                elif line.startswith("NEUPRINT_APPLICATION_CREDENTIALS="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")

    if not token:
        print("\n" + "="*70)
        print("ERROR: neuPrint API token not found!")
        print("To get your token:")
        print("1. Go to https://neuprint.janelia.org and log in with your Google account.")
        print("2. Click on your profile icon (top right) -> 'Account'.")
        print("3. Copy your Auth Token.")
        print("4. Provide it via:")
        print("   - Command line: python 01_pull_connectome.py --token <YOUR_TOKEN>")
        print("   - Environment variable: $env:NEUPRINT_TOKEN='<YOUR_TOKEN>'")
        print("   - Or save it in a '.env' file: NEUPRINT_TOKEN=<YOUR_TOKEN>")
        print("="*70 + "\n")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Connecting to {args.server} [{args.dataset}]...")
    client = Client(args.server, dataset=args.dataset, token=token)
    version = client.fetch_version()
    print(f"Connected successfully! Server version: {version}")

    print(f"\nFetching neurons for ROI='{args.roi}', status='Traced', min_pre={args.min_pre}...")
    neuron_df, roi_df = fetch_neurons(
        NC(status='Traced', rois=[args.roi], min_pre=args.min_pre)
    )
    print(f"Found {len(neuron_df)} matching neurons in {args.roi}.")

    # Select the first max_neurons
    neuron_ids = neuron_df['bodyId'].values[:args.max_neurons]
    selected_neuron_df = neuron_df[neuron_df['bodyId'].isin(neuron_ids)].copy()
    
    neuron_csv_path = os.path.join(args.output_dir, "neuron_df.csv")
    selected_neuron_df.to_csv(neuron_csv_path, index=False)
    print(f"Saved {len(neuron_ids)} selected neurons to {neuron_csv_path}")

    print(f"\nFetching adjacency matrix for {len(neuron_ids)} neurons...")
    _, conn_df = fetch_adjacencies(sources=neuron_ids, targets=neuron_ids)
    
    conn_csv_path = os.path.join(args.output_dir, "conn_df.csv")
    conn_df.to_csv(conn_csv_path, index=False)
    print(f"Saved {len(conn_df)} connections to {conn_csv_path}")

    # Build Weight Matrix W
    print("\nBuilding weight matrix W...")
    id_to_idx = {bid: i for i, bid in enumerate(neuron_ids)}
    N = len(neuron_ids)
    W = np.zeros((N, N), dtype=np.float32)

    for _, row in conn_df.iterrows():
        pre_id = row['bodyId_pre']
        post_id = row['bodyId_post']
        w = row['weight']
        if pre_id in id_to_idx and post_id in id_to_idx:
            i = id_to_idx[pre_id]
            j = id_to_idx[post_id]
            W[i, j] = w

    w_path = os.path.join(args.output_dir, "W.npy")
    np.save(w_path, W)
    print(f"Saved weight matrix to {w_path}")

    print("\nConnectome Statistics:")
    assert os.path.exists(w_path), f"Missing {w_path}"
    assert os.path.exists(neuron_csv_path), f"Missing {neuron_csv_path}"
    assert os.path.exists(conn_csv_path), f"Missing {conn_csv_path}"
    assert W.shape == (N, N), f"Expected shape ({N}, {N}), got {W.shape}"
    
    nonzero_edges = np.count_nonzero(W)
    density = nonzero_edges / (N * N) if N > 0 else 0
    max_weight = float(W.max()) if nonzero_edges > 0 else 0
    mean_weight = float(W[W > 0].mean()) if nonzero_edges > 0 else 0

    print(f"  Matrix Shape          : {W.shape}")
    print(f"  Neurons (N)           : {N}")
    print(f"  Synaptic Edges        : {nonzero_edges}")
    print(f"  Graph Density         : {density:.4%}")
    print(f"  Synaptic Weight Range : [{float(W.min()):.1f}, {max_weight:.1f}]")
    print(f"  Mean Non-zero Weight  : {mean_weight:.2f}")

    if nonzero_edges > 0:
        print(f"\nSubcircuit extraction complete. Saved weight matrix to {w_path}.")
    else:
        print("\nWarning: No synaptic connections found between selected neurons.")

if __name__ == "__main__":
    main()
