# Single-Choice DDM Simulation and Analysis Script
# This script simulates synthetic data for a single-choice Drift Diffusion Model (DDM),
# fits the model using PyDDM, generates predictions, and creates visualizations
# to illustrate the decision-making process, response time distributions,
# and model fit. It demonstrates the effect of coherence on detection probability
# and reaction times in a single-bound DDM framework.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyddm import Model, Fittable, Sample
from pyddm.models import Drift, NoiseConstant, BoundConstant
from pyddm.models import OverlayChain, OverlayNonDecision, OverlayPoissonMixture

# Defines drift rate based on coherence for single-choice DDM
class DriftCoherence(Drift):
    name = "Drift depending on coherence"
    required_parameters = ["drift_base", "drift_scale"]
    required_conditions = ["coherence"]

    def get_drift(self, x, t, conditions, **kwargs):
        coherence = conditions.get("coherence", 0.0)
        return self.drift_base + self.drift_scale * coherence

# Simulates single-bound DDM with upper bound only
def simulate_onebound(
    drift_base, drift_scale, noise, B, ndt,
    coherence, T_dur=5.0, dt=0.01, n_traj=20000, seed=123
):
    """
    Simulate detection probability and mean reaction time for a single-bound DDM.

    This function performs Monte Carlo simulations of evidence accumulation
    in a single-choice DDM with only an upper bound (detection threshold).
    Trials that do not cross the bound within the maximum duration are considered timeouts.

    Parameters:
    drift_base (float): Base drift rate.
    drift_scale (float): Scaling factor for coherence-dependent drift.
    noise (float): Noise level (standard deviation of Wiener process).
    B (float): Upper bound (detection threshold).
    ndt (float): Non-decision time.
    coherence (float): Coherence level for this simulation.
    T_dur (float, optional): Maximum trial duration. Defaults to 5.0 seconds.
    dt (float, optional): Time step. Defaults to 0.01 seconds.
    n_traj (int, optional): Number of trajectories to simulate. Defaults to 20000.
    seed (int, optional): Random seed for reproducibility. Defaults to 123.

    Returns:
    tuple: Detection probability and mean reaction time (including NDT).
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T_dur / dt)
    x = np.zeros(n_traj, dtype=float)
    t = np.zeros(n_traj, dtype=float)

    drift = drift_base + drift_scale * coherence
    hit = np.zeros(n_traj, dtype=bool)
    rt  = np.full(n_traj, T_dur + ndt, dtype=float)  # default: timeout

    for k in range(n_steps):
        alive = ~hit
        if not np.any(alive):
            break

        dW = rng.normal(loc=0.0, scale=np.sqrt(dt), size=alive.sum())
        x[alive] += drift * dt + noise * dW
        t[alive] += dt

        crossed = alive.copy()
        crossed[alive] = x[alive] >= B
        if np.any(crossed):
            hit[crossed] = True
            rt[crossed]  = t[crossed] + ndt

    p_detected = hit.mean()
    mean_rt = rt.mean()
    return p_detected, mean_rt

# Generates sample single-choice DDM trajectories for visualization
def generate_single_choice_trajectories(drift, noise, B, n_trajectories=50, T_dur=3.0, dt=0.01, seed=123):
    """
    Generate evidence accumulation trajectories for single-choice DDM visualization.

    This function simulates multiple trajectories until they reach the detection bound
    or the maximum time, clamping the evidence at the bound after crossing.

    Parameters:
    drift (float): Drift rate.
    noise (float): Noise level.
    B (float): Detection bound.
    n_trajectories (int, optional): Number of trajectories. Defaults to 50.
    T_dur (float, optional): Maximum duration. Defaults to 3.0 seconds.
    dt (float, optional): Time step. Defaults to 0.01 seconds.
    seed (int, optional): Random seed. Defaults to 123.

    Returns:
    tuple: List of trajectories, path tuples (time, evidence), decision times, and decisions (1 for detected, 0 for miss).
    """
    np.random.seed(seed)
    n_steps = int(T_dur / dt)
    time_points = np.arange(0, T_dur, dt)[:n_steps]
    
    trajectories = []
    decision_times = []
    decisions = []
    trajectory_paths = []
    
    for i in range(n_trajectories):
        x = np.zeros(n_steps)
        x[0] = 0
        
        detected = False
        decision_time = T_dur
        decision_idx = n_steps - 1
        
        for t_idx in range(1, n_steps):
            dW = np.random.randn() * np.sqrt(dt)
            x[t_idx] = x[t_idx-1] + drift * dt + noise * dW
            
            if not detected and x[t_idx] >= B:
                detected = True
                decision_time = time_points[t_idx]
                decision_idx = t_idx
                x[t_idx:] = B
                break
        
        if detected:
            trajectory_path = x[:decision_idx+1]
            time_path = time_points[:decision_idx+1]
            decision = 1
        else:
            trajectory_path = x
            time_path = time_points
            decision = 0
        
        trajectories.append(trajectory_path)
        trajectory_paths.append((time_path, trajectory_path))
        decision_times.append(decision_time)
        decisions.append(decision)
    
    return trajectories, trajectory_paths, decision_times, decisions

# Creates a visualization of single-choice DDM trajectories and RT distributions
def create_single_choice_ddm_visualization(drift_base, drift_scale, noise, B, ndt, coherence_levels):
    """
    Create a multi-panel figure visualizing DDM trajectories and RT distributions for different coherence levels.

    The top row shows evidence accumulation trajectories, and the bottom row shows
    reaction time distributions for detected trials, with indicators for non-decision time and timeouts.

    Parameters:
    drift_base (float): Base drift rate.
    drift_scale (float): Coherence scaling for drift.
    noise (float): Noise level.
    B (float): Bound.
    ndt (float): Non-decision time.
    coherence_levels (list): List of coherence values to visualize.

    Returns:
    matplotlib.figure.Figure: The created figure.
    """
    print(f"  Creating single-choice DDM process visualization")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Single-Choice DDM: Process Visualization", fontsize=16)
    
    for i, coherence in enumerate(coherence_levels):
        drift = drift_base + drift_scale * coherence
        
        trajectories, trajectory_paths, decision_times, decisions = generate_single_choice_trajectories(
            drift=drift, noise=noise, B=B, n_trajectories=60, T_dur=3.0, seed=42 + i*100
        )
        
        ax_traj = axes[0, i]
        ax_traj.axhline(y=B, color='green', linestyle='-', linewidth=3, 
                       label=f'Detection Threshold (+{B:.1f})', alpha=0.8)
        ax_traj.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='Starting Point')
        
        detected_count = sum(1 for d in decisions if d == 1)
        miss_count = sum(1 for d in decisions if d == 0)
        
        for j, (time_path, traj_path) in enumerate(trajectory_paths):
            color = 'green' if decisions[j] == 1 else 'red'
            alpha = 0.7 if decisions[j] == 1 else 0.5
            linewidth = 1.8 if decisions[j] == 1 else 1.2
            ax_traj.plot(time_path, traj_path, color=color, alpha=alpha, linewidth=linewidth)
        
        ax_traj.annotate(f'Drift Rate = {drift:.2f}', 
                        xy=(2.2, B*0.3), fontsize=11, ha='center', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcyan", alpha=0.8))
        
        ax_traj.set_title(f'Coherence = {coherence:.1f}\n(Detected: {detected_count}, Miss: {miss_count})')
        ax_traj.set_xlabel('Time (s)')
        ax_traj.set_ylabel('Evidence (Δx)')
        ax_traj.set_ylim(-0.5, B*1.3)
        ax_traj.set_xlim(0, 3.0)
        ax_traj.grid(True, alpha=0.3)
        ax_traj.legend(loc='upper left', fontsize=9)
        
        ax_rt = axes[1, i]
        detected_rts = [dt for dt, dec in zip(decision_times, decisions) if dec == 1]
        
        if detected_rts:
            bins = np.linspace(min(detected_rts), max(detected_rts), 15)
            ax_rt.hist(detected_rts, bins=bins, alpha=0.7, color='green', 
                      label=f'Detected (n={len(detected_rts)})', density=True)
            mean_detected_rt = np.mean(detected_rts)
            ax_rt.axvline(x=mean_detected_rt, color='darkgreen', linestyle=':', linewidth=2,
                         label=f'Mean detected RT ({mean_detected_rt:.3f}s)')
        
        ax_rt.axvline(x=ndt, color='purple', linestyle='--', linewidth=3, alpha=0.8,
                     label=f'Non-decision time ({ndt:.3f}s)')
        timeout_line = 3.0 + ndt
        ax_rt.axvline(x=timeout_line, color='red', linestyle='-', linewidth=2, alpha=0.6,
                     label=f'Timeout ({timeout_line:.1f}s)\n{miss_count} miss trials')
        
        ax_rt.set_title(f'RT Distribution\nCoherence = {coherence:.1f}')
        ax_rt.set_xlabel('Reaction Time (s)')
        ax_rt.set_ylabel('Density')
        ax_rt.legend(fontsize=8)
        ax_rt.grid(True, alpha=0.3)
        ax_rt.set_xlim(0.1, 3.5)
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    # Data Generation
    print(f"\nData Generation:")
    print(f"  Generating synthetic single-choice behavioral data")

    coherences = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_trials_per_cond = 300
    true_drift_base = -0.8
    true_drift_scale = 2.0
    true_noise = 1.0
    true_bound = 1.2
    true_ndt = 0.2
    max_time = 3.0

    np.random.seed(42)
    rts, choices, coh_vals = [], [], []

    for coh in coherences:
        for _ in range(n_trials_per_cond):
            drift = true_drift_base + true_drift_scale * coh
            x = 0.0
            t = 0.0
            detected = False

            while t < max_time:
                x += drift * 0.01 + true_noise * np.sqrt(0.01) * np.random.randn()
                t += 0.01
                if x >= true_bound:
                    rt = t + true_ndt
                    rts.append(rt)
                    choices.append(1)
                    coh_vals.append(coh)
                    detected = True
                    break

            if not detected:
                rts.append(max_time + true_ndt)
                choices.append(0)
                coh_vals.append(coh)

    data = pd.DataFrame({"rt": rts, "detected": choices, "coherence": coh_vals})
    data = data[(data["rt"] >= 0.1) & (data["rt"] <= 3.5)].copy()

    sample_data = Sample.from_pandas_dataframe(
        data,
        "rt", "detected", ("not_detected", "detected"), "coherence"
    )
    
    print(f"\nData Summary:")
    print(f"  Generated {len(data)} trials across {len(coherences)} coherence levels")
    
    print(f"\nDetection Rate by Coherence:")
    for coh in coherences:
        det_rate = data[data["coherence"] == coh]["detected"].mean()
        mean_rt = data[data["coherence"] == coh]["rt"].mean()
        print(f"  Coherence {coh:.1f}: {det_rate:.3f} detection, {mean_rt:.3f}s RT")

    # Model Definition
    print(f"\nModel Definition:")
    print(f"  Defining single-choice DDM model")

    model = Model(
        name="Drift ~ Coherence (fit), one-bound predictions",
        drift=DriftCoherence(
            drift_base=Fittable(minval=-2.0, maxval=2.0, value=0.0),
            drift_scale=Fittable(minval=0.0, maxval=6.0, value=2.0)
        ),
        noise=NoiseConstant(noise=true_noise),
        bound=BoundConstant(B=Fittable(minval=0.5, maxval=4.0, value=1.2)),
        overlay=OverlayChain(overlays=[
            OverlayNonDecision(nondectime=Fittable(minval=0.0, maxval=0.5, value=0.2)),
            OverlayPoissonMixture(
                pmixturecoef=Fittable(minval=0.0, maxval=0.2, value=0.05),
                rate=Fittable(minval=0.5, maxval=5.0, value=1.0)
            )
        ]),
        dx=0.01, dt=0.01, T_dur=3.5,
        choice_names=("not_detected", "detected")
    )

    # Model Fitting
    print(f"\nModel Fitting:")
    print(f"  Fitting model to data")
    
    try:
        from pyddm.functions import LossRobustBIC
        model.fit(sample_data, lossfunction=LossRobustBIC, verbose=False)
        print(f"  Model fitting completed")
        fitting_successful = True
    except ImportError:
        try:
            from pyddm.functions import LossRobustLikelihood
            model.fit(sample_data, lossfunction=LossRobustLikelihood, verbose=False)
            print(f"  Model fitting completed")
            fitting_successful = True
        except ImportError:
            print(f"  Robust likelihood not available, using default fitting")
            try:
                model.fit(sample_data, verbose=False)
                print(f"  Model fitting completed")
                fitting_successful = True
            except Exception as e:
                print(f"  Fitting error: {str(e)}")
                print(f"  Using initial parameter values")
                fitting_successful = False

    print(f"\nFitted Parameters:")
    fitted = {}
    try:
        if fitting_successful:
            for name, val in zip(model.get_model_parameter_names(), model.get_model_parameters()):
                print(f"  {name}: {val:.4f}")
                fitted[name] = val
        else:
            raise Exception("Fitting failed")
    except:
        print(f"  Using default parameter values")
        fitted = {
            "drift_base": -0.8,
            "drift_scale": 2.0,
            "B": 1.2,
            "nondectime": 0.2
        }

    drift_base = fitted.get("drift_base", -0.8)
    drift_scale = fitted.get("drift_scale", 2.0)
    B = fitted.get("B", 1.2)
    ndt = fitted.get("nondectime", 0.2)
    noise = true_noise

    # Model Predictions
    print(f"\nModel Predictions:")
    print(f"  Generating simulation-based predictions")

    coherence_bins = sorted(data["coherence"].unique())
    mean_det_data = data.groupby("coherence")["detected"].mean().values
    mean_rt_data  = data.groupby("coherence")["rt"].mean().values

    pred_det, pred_rt = [], []
    for coh in coherence_bins:
        p_hit, mrt = simulate_onebound(
            drift_base=drift_base, drift_scale=drift_scale,
            noise=noise, B=B, ndt=ndt,
            coherence=coh, T_dur=3.5, dt=model.dt,
            n_traj=30000, seed=123 + int(coh*1000)
        )
        pred_det.append(p_hit)
        pred_rt.append(mrt)
        print(f"    Coherence={coh:.1f}: detection={p_hit:.3f}, mean_rt={mrt:.3f}s")

    # Validation Checks
    print(f"\nValidation Checks:")
    sim_sensible = pred_det[-1] > pred_det[0]
    sim_varied = np.std(pred_det) > 0.05
    print(f"  Simulation sensible predictions: {sim_sensible}")
    print(f"  Simulation varied predictions: {sim_varied}")

    # Visualization
    print(f"\nVisualization:")
    print(f"  Creating main analysis figure")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Single-Choice DDM: Model Results", fontsize=14)

    ax = axes[0, 0]
    detected_rts = data[data["detected"] == 1]["rt"]
    ax.hist(detected_rts, bins=30, density=True, alpha=0.7, label=f"Detected (n={len(detected_rts)})", color="green")
    n_misses = len(data[data["detected"] == 0])
    ax.set_title(f"RT Distribution - Detected Trials\n(n={n_misses} miss trials)")
    ax.set_xlabel("RT (s)")
    ax.set_ylabel("Density")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(coherence_bins, mean_det_data, "o-", label="Data", linewidth=2, markersize=8, color="blue")
    ax.plot(coherence_bins, pred_det, "s--", label="Simulation", linewidth=2, markersize=6, color="orange")
    ax.set_ylim(0, 1.05)
    ax.set_title("Detection Probability vs Coherence")
    ax.set_xlabel("Coherence")
    ax.set_ylabel("Detection Probability")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(coherence_bins, mean_rt_data, "o-", label="Data", linewidth=2, markersize=8, color="blue")
    ax.plot(coherence_bins, pred_rt, "s--", label="Simulation", linewidth=2, markersize=6, color="orange")
    ax.set_title("Mean RT vs Coherence")
    ax.set_xlabel("Coherence")
    ax.set_ylabel("Mean RT (s)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(coherence_bins)))
    for i, coh in enumerate(coherence_bins):
        coh_data = data[data["coherence"] == coh]
        mean_rt_coh = coh_data["rt"].mean()
        mean_det_coh = coh_data["detected"].mean()
        ax.scatter(mean_rt_coh, mean_det_coh, color=colors[i], s=100, 
                  label=f"Coh {coh:.1f}")
    ax.set_title("Speed-Detection Tradeoff")
    ax.set_xlabel("Mean RT (s)")
    ax.set_ylabel("Detection Probability")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    
    print(f"  Creating RT distribution plots by coherence")
    
    fig2, axes2 = plt.subplots(1, len(coherence_bins), figsize=(16, 6), sharey=True)
    fig2.suptitle("RT Distributions by Coherence: Detected Responses", fontsize=14)

    for i, coh in enumerate(coherence_bins):
        coh_data = data[data["coherence"] == coh]
        detected_trials = coh_data[coh_data["detected"] == 1]["rt"]
        n_detected = len(detected_trials)
        n_missed = len(coh_data[coh_data["detected"] == 0])
        
        ax2 = axes2[i]
        if len(detected_trials) > 0:
            ax2.hist(detected_trials, bins=20, density=True, alpha=0.7, color="green")
            ax2.axvline(detected_trials.mean(), color="darkgreen", linestyle="--", 
                       label=f"Mean: {detected_trials.mean():.3f}s")
        ax2.set_title(f"Coh={coh:.1f}\nDetected (n={n_detected}, {n_missed} miss)")
        ax2.set_xlabel("RT (s)")
        if i == 0:
            ax2.set_ylabel("Density")
        if len(detected_trials) > 0:
            ax2.legend(fontsize=8)

    plt.tight_layout()

    print(f"  Creating single-choice DDM process visualization")
    
    vis_coherences = [0.1, 0.5, 0.9]
    fig3 = create_single_choice_ddm_visualization(
        drift_base=drift_base, 
        drift_scale=drift_scale, 
        noise=noise, 
        B=B, 
        ndt=ndt, 
        coherence_levels=vis_coherences
    )
    plt.tight_layout()

    print(f"  Displaying all figures")
    plt.show(block=True)

    print(f"\nAnalysis Summary:")
    print(f"  Single-choice DDM analysis completed")
    print(f"  Detection range: {min(mean_det_data):.3f} to {max(mean_det_data):.3f}")
    print(f"  RT range: {min(mean_rt_data):.3f}s to {max(mean_rt_data):.3f}s")
    print(f"  Model captures detection curve: {pred_det[-1] > pred_det[0]}")
    print(f"  Low coherence has substantial misses: {mean_det_data[0] < 0.6}")
    print(f"  High coherence has high detection: {mean_det_data[-1] > 0.8}")

    input("Press Enter to close all figures...")
    plt.close('all')