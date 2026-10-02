import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from brian2 import (
    ms, mV, volt, second,
    prefs, start_scope, NeuronGroup, Synapses, SpikeMonitor, StateMonitor, Network, defaultclock
)

# Use numpy backend for reliable execution across all OS environments
prefs.codegen.target = 'numpy'

def build_lif_network(
    W,
    tau=20 * ms,
    v_rest=-65 * mV,
    v_thresh=-50 * mV,
    v_reset=-65 * mV,
    synapse_scale=0.05,
    synapse_clip=5.0 * mV,
    record_states=True,
    num_state_neurons=10,
    dt=0.1 * ms
):
    """
    Builds a Brian2 Leaky Integrate-and-Fire (LIF) network wired according to
    the biological connectome weight matrix W.

    Args:
        W: 2D numpy array of shape (N, N) representing synaptic weights.
        tau: Membrane time constant.
        v_rest: Resting membrane potential.
        v_thresh: Action potential threshold.
        v_reset: Reset potential post-spike.
        synapse_scale: Scaling factor mapping synapse count to mV PSP.
        synapse_clip: Maximum individual synaptic weight (prevents runaway seizure).
        record_states: Whether to record voltage traces for a subset of neurons.
        num_state_neurons: Number of neurons to record voltage from.
        dt: Simulation time step.

    Returns:
        net: Brian2 Network object containing all components.
        G: NeuronGroup.
        S: Synapses.
        spike_mon: SpikeMonitor.
        state_mon: StateMonitor (or None if record_states is False).
    """
    start_scope()
    defaultclock.dt = dt

    N = W.shape[0]

    # Leaky Integrate-and-Fire differential equation
    eqs = '''
    dv/dt = (v_rest - v + I_ext) / tau : volt
    I_ext : volt
    '''

    ns = {
        'v_rest': v_rest,
        'tau': tau,
        'v_thresh': v_thresh,
        'v_reset': v_reset
    }

    G = NeuronGroup(
        N,
        eqs,
        threshold='v > v_thresh',
        reset='v = v_reset',
        method='exact',
        namespace=ns
    )
    G.v = v_rest
    G.I_ext = 0 * mV

    # Connectome synapses
    pre_idx, post_idx = np.nonzero(W)
    S = Synapses(G, G, 'w : volt', on_pre='v_post += w')
    S.connect(i=pre_idx, j=post_idx)

    # Scale and clip synaptic weights
    w_unclipped = W[pre_idx, post_idx] * synapse_scale
    if synapse_clip is not None:
        w_clipped = np.clip(w_unclipped, 0, synapse_clip / mV)
    else:
        w_clipped = w_unclipped
    S.w = w_clipped * mV

    # Monitors
    spike_mon = SpikeMonitor(G)
    state_mon = None
    if record_states:
        record_indices = list(range(min(num_state_neurons, N)))
        state_mon = StateMonitor(G, 'v', record=record_indices)
        net = Network(G, S, spike_mon, state_mon)
    else:
        net = Network(G, S, spike_mon)

    return net, G, S, spike_mon, state_mon


def run_calibration_test(
    weight_matrix_path="outputs/W.npy",
    output_fig_path="outputs/figures/lif_simulation_stage2.png",
    sim_duration=100 * ms,
    n_input_neurons=50,
    synapse_scale=0.05,
    synapse_clip=5.0 * mV,
):
    print("Initializing Leaky Integrate-and-Fire Network...")

    assert os.path.exists(weight_matrix_path), f"Connectome weight matrix not found at {weight_matrix_path}. Run Stage 1 first."
    W = np.load(weight_matrix_path)
    N = W.shape[0]
    print(f"Loaded connectome weight matrix: shape {W.shape} ({N} neurons, {np.count_nonzero(W)} synapses)")

    print(f"Building LIF network...")
    print(f"  tau           : 20 ms")
    print(f"  v_rest        : -65 mV")
    print(f"  v_thresh      : -50 mV")
    print(f"  v_reset       : -65 mV")
    print(f"  synapse_scale : {synapse_scale}")
    print(f"  synapse_clip  : {synapse_clip}")
    print(f"  input neurons : {n_input_neurons}")

    net, G, S, spike_mon, state_mon = build_lif_network(
        W=W,
        synapse_scale=synapse_scale,
        synapse_clip=synapse_clip,
        record_states=True,
        num_state_neurons=10
    )

    # Inject test graded currents into first 50 neurons
    # Values between 0 and 25 mV simulate an MNIST digit stroke
    np.random.seed(42)
    test_inputs = np.random.uniform(0, 25, n_input_neurons)
    G.I_ext = np.zeros(N) * mV
    G.I_ext[:n_input_neurons] = test_inputs * mV

    print(f"\nRunning simulation for {sim_duration}...")
    net.run(sim_duration)

    # Analyze spiking activity
    spike_train_i = np.array(spike_mon.i)
    spike_train_t = np.array(spike_mon.t / ms)
    total_spikes = len(spike_train_i)
    active_neurons = len(np.unique(spike_train_i))

    input_spikes = int(np.sum(spike_train_i < n_input_neurons))
    downstream_spikes = int(np.sum(spike_train_i >= n_input_neurons))
    active_downstream = len(np.unique(spike_train_i[spike_train_i >= n_input_neurons]))

    print("\n--- Simulation Results ---")
    print(f"Total Spikes Fired            : {total_spikes}")
    print(f"Total Active Neurons          : {active_neurons} / {N} ({active_neurons/N:.1%})")
    print(f"Input Neuron Spikes (0..{n_input_neurons-1})   : {input_spikes}")
    print(f"Downstream Spikes ({n_input_neurons}..{N-1})   : {downstream_spikes}")
    print(f"Active Downstream Neurons     : {active_downstream} / {N - n_input_neurons}")

    # Generate Stage 2 Diagnostic Figure
    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False, gridspec_kw={'height_ratios': [2, 1.2]})

    # Subplot 1: Spike Raster Plot
    ax_raster = axes[0]
    if total_spikes > 0:
        # Input neurons in orange, downstream in deep blue
        is_input = spike_train_i < n_input_neurons
        ax_raster.scatter(
            spike_train_t[is_input], spike_train_i[is_input],
            c='#FF5722', s=16, alpha=0.8, label=f'Input Neurons (0–{n_input_neurons-1})'
        )
        ax_raster.scatter(
            spike_train_t[~is_input], spike_train_i[~is_input],
            c='#1976D2', s=16, alpha=0.8, label=f'Downstream Circuit ({n_input_neurons}–{N-1})'
        )
    ax_raster.axhline(n_input_neurons - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax_raster.set_ylabel('Neuron Index', fontsize=11)
    ax_raster.set_ylim(-5, N + 5)
    ax_raster.set_xlim(0, float(sim_duration / ms))
    ax_raster.set_title('Stage 2: Spike Raster Across Connectome Reservoir (100 ms Trial)', fontsize=12, fontweight='bold')
    ax_raster.legend(loc='upper right', framealpha=0.9)
    ax_raster.grid(True, linestyle=':', alpha=0.5)

    # Subplot 2: Membrane Voltage Traces
    ax_v = axes[1]
    times_ms = state_mon.t / ms
    for i in range(min(5, len(state_mon.record))):
        v_trace = state_mon.v[i] / mV
        ax_v.plot(times_ms, v_trace, label=f'Neuron {i}', alpha=0.85, linewidth=1.2)
    ax_v.axhline(-50, color='red', linestyle='--', linewidth=1, label=r'Threshold $v_{thresh} = -50$ mV')
    ax_v.axhline(-65, color='black', linestyle=':', linewidth=1, label=r'Resting $v_{rest} = -65$ mV')
    ax_v.set_xlabel('Time (ms)', fontsize=11)
    ax_v.set_ylabel('Membrane Potential (mV)', fontsize=11)
    ax_v.set_xlim(0, float(sim_duration / ms))
    ax_v.set_title('Membrane Potential Dynamics $v(t)$', fontsize=12, fontweight='bold')
    ax_v.legend(loc='lower right', ncol=3, fontsize=9, framealpha=0.9)
    ax_v.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_fig_path, dpi=150)
    plt.close()
    print(f"Saved diagnostic raster & voltage figure to: {output_fig_path}")

    assert total_spikes > 0, "No spikes generated during simulation"
    assert input_spikes > 0, "Input neurons did not fire"
    assert downstream_spikes > 0, "Signal did not propagate downstream"
    assert os.path.exists(output_fig_path), f"Figure not found at {output_fig_path}"
    print("Simulation and calibration complete.")

    return {
        "total_spikes": total_spikes,
        "active_neurons": active_neurons,
        "input_spikes": input_spikes,
        "downstream_spikes": downstream_spikes,
        "active_downstream": active_downstream
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: LIF Network Simulation & Calibration")
    parser.add_argument("--weights", type=str, default="outputs/W.npy", help="Path to W.npy")
    parser.add_argument("--fig-path", type=str, default="outputs/figures/lif_simulation_stage2.png", help="Path to save figure")
    parser.add_argument("--scale", type=float, default=0.05, help="Synapse weight scale")
    parser.add_argument("--clip", type=float, default=5.0, help="Synapse max clip (mV)")
    args = parser.parse_args()

    run_calibration_test(
        weight_matrix_path=args.weights,
        output_fig_path=args.fig_path,
        synapse_scale=args.scale,
        synapse_clip=args.clip * mV
    )
