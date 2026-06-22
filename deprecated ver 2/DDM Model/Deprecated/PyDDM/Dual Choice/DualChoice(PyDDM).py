# Dual-Choice PyDDM Analysis Script
# This script performs a Drift Diffusion Model (DDM) analysis for dual-choice tasks using PyDDM.
# It loads and validates data from a CSV file, fits a DDM model, simulates decision trajectories,
# and generates comprehensive visualizations to illustrate the results.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pyddm import Model, Fittable, Sample
from pyddm.models import DriftConstant, NoiseConstant, BoundConstant
from pyddm.models import OverlayChain, OverlayNonDecision, OverlayPoissonMixture

# Suppress PyDDM warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pyddm')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Try to import robust fitting methods
try:
    from pyddm.functions import LossRobustBIC
    ROBUST_AVAILABLE = True
except ImportError:
    ROBUST_AVAILABLE = False

def load_csv_for_pyddm(file_path):
    """
    Load and prepare CSV data for PyDDM analysis.

    This function reads a CSV file into a pandas DataFrame, converts relevant columns to numeric types,
    removes invalid trials (e.g., missing or out-of-range values), and provides a summary of the cleaned data.

    Parameters:
    file_path (str): The path to the CSV file containing the DDM data.

    Returns:
    pd.DataFrame: The cleaned DataFrame ready for PyDDM analysis.
    """
    # Load data
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} trials")
    
    # Convert to numeric and clean data
    df['response_time'] = pd.to_numeric(df['response_time'], errors='coerce')
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    
    # Remove invalid trials
    df = df.dropna(subset=['response_time', 'accuracy'])
    df = df[(df['response_time'] > 0) & (df['response_time'] < 10)]
    df = df[df['accuracy'].isin([0, 1])]
    
    print(f"After cleaning: {len(df)} trials")
    print(f"Accuracy: {df['accuracy'].mean():.3f}")
    print(f"Mean RT: {df['response_time'].mean():.3f}s")
    
    return df

def fit_pyddm_model(df):
    """
    Fit a PyDDM model to the dual-choice data.

    This function prepares a PyDDM Sample from the DataFrame, defines a DDM model with constant drift,
    noise, and bounds, and fits the model using either robust or standard methods. It handles errors
    during sample creation and fitting, returning the fitted model and parameters or defaults if fitting fails.

    Parameters:
    df (pd.DataFrame): The cleaned DataFrame containing trial data.

    Returns:
    tuple: The fitted PyDDM Model (or None if fitting fails) and a dictionary of fitted parameters.
    """
    # Prepare data for PyDDM
    df_sample = df.copy()
    df_sample['choice'] = df_sample['accuracy']  # 0=incorrect, 1=correct
    
    # Create PyDDM sample
    try:
        sample = Sample.from_pandas_dataframe(
            df_sample,
            rt_column_name="response_time",
            choice_column_name="choice",
            choice_names=("incorrect", "correct")
        )
        print(f"Created PyDDM sample with {len(sample)} trials")
    except Exception as e:
        print(f"Error creating sample: {e}")
        return None, {}
    
    # Define model with mixture overlay to handle outliers
    model = Model(
        name="DDM with Mixture",
        drift=DriftConstant(drift=Fittable(minval=-3, maxval=3, value=1)),
        noise=NoiseConstant(noise=1),
        bound=BoundConstant(B=Fittable(minval=0.3, maxval=3, value=1)),
        overlay=OverlayChain(overlays=[
            OverlayNonDecision(nondectime=Fittable(minval=0, maxval=0.8, value=0.2)),
            OverlayPoissonMixture(
                pmixturecoef=Fittable(minval=0.0, maxval=0.2, value=0.05),
                rate=Fittable(minval=0.1, maxval=2.0, value=1.0)
            )
        ]),
        dx=0.01, dt=0.01, T_dur=5.0,
        choice_names=("incorrect", "correct")
    )
    
    # Fit model with error suppression
    try:
        print("Fitting model...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if ROBUST_AVAILABLE:
                model.fit(sample, lossfunction=LossRobustBIC, verbose=False)
                print("Model fitted successfully (robust method)!")
            else:
                model.fit(sample, verbose=False)
                print("Model fitted successfully (standard method)!")
        
        # Extract parameters
        param_names = model.get_model_parameter_names()
        param_values = model.get_model_parameters()
        fitted_params = dict(zip(param_names, param_values))
        
        print("\nFitted Parameters:")
        for name, value in fitted_params.items():
            print(f"  {name}: {value:.4f}")
            
        return model, fitted_params
        
    except Exception as e:
        print(f"Error fitting model: {e}")
        return None, {}

def simulate_ddm_trajectories(drift, boundary, ndt, noise=1.0, n_trajectories=50, dt=0.01, max_time=3.0):
    """
    Simulates evidence accumulation trajectories for a dual-choice DDM.

    This function generates multiple simulated trajectories based on the provided DDM parameters.
    Each trajectory accumulates evidence over time until it hits an upper (correct) or lower (incorrect)
    boundary or reaches the maximum time, with a forced choice based on evidence sign at timeout.

    Parameters:
    drift (float): The drift rate parameter.
    boundary (float): The absolute value of the decision boundaries (symmetric).
    ndt (float): The non-decision time.
    noise (float, optional): The noise level. Defaults to 1.0.
    n_trajectories (int, optional): Number of trajectories to simulate. Defaults to 50.
    dt (float, optional): Time step for simulation. Defaults to 0.01 seconds.
    max_time (float, optional): Maximum simulation time per trajectory. Defaults to 3.0 seconds.

    Returns:
    list: A list of dictionaries, each containing 'time', 'evidence', 'choice', 'rt', and 'decided' for a trajectory.
    """
    np.random.seed(42)
    trajectories = []
    
    for i in range(n_trajectories):
        time_points = [0]
        evidence = [0]
        t = 0
        x = 0
        decided = False
        choice = None
        
        while t < max_time and not decided:
            # Update evidence
            dW = np.random.randn() * np.sqrt(dt)
            x += drift * dt + noise * dW
            t += dt
            
            time_points.append(t)
            evidence.append(x)
            
            # Check boundaries
            if x >= boundary:
                choice = 1  # correct
                decided = True
            elif x <= -boundary:
                choice = 0  # incorrect  
                decided = True
        
        if not decided:
            choice = 1 if x > 0 else 0  # forced choice
        
        rt = t + ndt
        
        trajectories.append({
            'time': time_points,
            'evidence': evidence,
            'choice': choice,
            'rt': rt,
            'decided': decided
        })
    
    return trajectories

def plot_comprehensive_results(df, model, params):
    """
    Creates comprehensive visualizations of the dual-choice DDM analysis results.

    This function generates a 2x3 grid of plots showing response time distributions,
    overall accuracy, response times by accuracy, fitted parameters, simulated trajectories,
    and a text summary of key statistics. It handles cases where model fitting fails
    by displaying placeholder text in the trajectory plot.

    Parameters:
    df (pd.DataFrame): The cleaned DataFrame with trial data.
    model (pyddm.Model or None): The fitted PyDDM model.
    params (dict): Dictionary of fitted parameters.

    Returns:
    None: Displays the figure using matplotlib.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('PyDDM Analysis Results', fontsize=16, fontweight='bold')
    
    # Plot 1: RT distributions
    ax = axes[0, 0]
    correct_trials = df[df['accuracy'] == 1]['response_time']
    incorrect_trials = df[df['accuracy'] == 0]['response_time']
    
    ax.hist(df['response_time'], bins=30, alpha=0.5, color='lightblue', 
           density=True, label=f'All trials (n={len(df)})')
    if len(correct_trials) > 10:
        ax.hist(correct_trials, bins=25, alpha=0.7, color='green', 
               density=True, label=f'Correct (n={len(correct_trials)})')
    if len(incorrect_trials) > 10:
        ax.hist(incorrect_trials, bins=25, alpha=0.7, color='red', 
               density=True, label=f'Incorrect (n={len(incorrect_trials)})')
    
    ax.set_xlabel('Response Time (s)')
    ax.set_ylabel('Density')
    ax.set_title('Response Time Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy summary
    ax = axes[0, 1]
    ax.bar(['Overall'], [df['accuracy'].mean()], color='skyblue', alpha=0.7)
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Overall Accuracy: {df["accuracy"].mean():.3f}')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: RT by Accuracy
    ax = axes[0, 2]
    correct_rt = df[df['accuracy'] == 1]['response_time'].mean() if len(correct_trials) > 0 else 0
    incorrect_rt = df[df['accuracy'] == 0]['response_time'].mean() if len(incorrect_trials) > 0 else 0
    ax.bar(['Correct', 'Incorrect'], [correct_rt, incorrect_rt], 
           color=['green', 'red'], alpha=0.7)
    ax.set_ylabel('Mean RT (s)')
    ax.set_title('Response Time by Accuracy')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Parameter values
    ax = axes[1, 0]
    param_names = ['Drift', 'Boundary', 'Non-Dec Time', 'Mix Coef']
    param_keys = ['drift', 'B', 'nondectime', 'pmixturecoef']
    param_values = [params.get(key, 0) for key in param_keys]
    colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
    
    bars = ax.bar(param_names, param_values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Parameter Value')
    ax.set_title('Fitted DDM Parameters')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, param_values):
        if not np.isnan(value) and value != 0:
            ax.text(bar.get_x() + bar.get_width()/2, 
                   bar.get_height() + max([v for v in param_values if v > 0]) * 0.02, 
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: Model Trajectories
    ax = axes[1, 1]
    if params:
        drift = params.get('drift', 0.5)
        boundary = params.get('B', 1.0)
        ndt = params.get('nondectime', 0.2)
        
        trajectories = simulate_ddm_trajectories(drift, boundary, ndt, n_trajectories=30)
        
        ax.axhline(boundary, color='green', linewidth=2, alpha=0.8, label='Upper Boundary')
        ax.axhline(-boundary, color='red', linewidth=2, alpha=0.8, label='Lower Boundary')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Starting Point')
        
        for traj in trajectories:
            color = 'green' if traj['choice'] == 1 else 'red'
            alpha = 0.6 if traj['decided'] else 0.3
            ax.plot(traj['time'], traj['evidence'], color=color, alpha=alpha, linewidth=1)
        
        ax.set_xlim(0, 2)
        ax.set_ylim(-boundary*1.2, boundary*1.2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Evidence')
        ax.set_title('Simulated DDM Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No fitted parameters\navailable', 
               transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title('DDM Trajectories')
    
    # Plot 6: Summary Statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    n_trials = len(df)
    overall_accuracy = df['accuracy'].mean()
    mean_rt = df['response_time'].mean()
    mean_correct_rt = df[df['accuracy'] == 1]['response_time'].mean() if len(correct_trials) > 0 else 0
    mean_incorrect_rt = df[df['accuracy'] == 0]['response_time'].mean() if len(incorrect_trials) > 0 else 0
    
    summary_text = f"""ANALYSIS SUMMARY

Dataset:
• Total trials: {n_trials:,}
• Overall accuracy: {overall_accuracy:.3f}

Response Times:
• Mean RT: {mean_rt:.3f}s
• Correct RT: {mean_correct_rt:.3f}s
• Incorrect RT: {mean_incorrect_rt:.3f}s

Model Parameters:
• Drift rate: {params.get('drift', 'N/A'):.3f}
• Boundary: {params.get('B', 'N/A'):.3f}
• Non-decision time: {params.get('nondectime', 'N/A'):.3f}s
• Mixture coef: {params.get('pmixturecoef', 'N/A'):.3f}

Model Quality:
• Fitting: {'Success' if params else 'Failed'}
• Method: {'Robust' if ROBUST_AVAILABLE else 'Standard'}
"""
    
    ax.text(0.05, 0.95, summary_text.strip(), transform=ax.transAxes,
           verticalalignment='top', fontsize=10, fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def analyze_csv(file_path):
    """
    Main function to perform dual-choice PyDDM analysis on a CSV file.

    This function orchestrates the entire analysis: loading data, fitting the model,
    and generating visualizations. It provides console output for progress and results,
    and handles cases where model fitting fails by returning None and an empty parameter dictionary.

    Parameters:
    file_path (str): The path to the CSV file containing the data.

    Returns:
    tuple: The fitted model (or None) and a dictionary of parameters.
    """
    print("=" * 50)
    print("Simplified PyDDM Analysis")
    print("=" * 50)
    
    # Load data
    df = load_csv_for_pyddm(file_path)
    
    # Fit model
    model, params = fit_pyddm_model(df)
    
    if model is not None:
        print("\n" + "=" * 50)
        print("Analysis Complete!")
        print("=" * 50)
        
        # Create comprehensive plots
        plot_comprehensive_results(df, model, params)
        
        return model, params
    else:
        print("Analysis failed.")
        return None, {}

# Main execution
if __name__ == "__main__":
    file_path = r"C:\Users\Rishv\OneDrive\Documents\Personal\Projects\PyDDM\Dual Choice\Data\dual_choice_ddm_extended.csv"
    
    print("Expected CSV format:")
    print("- response_time: reaction times in seconds")
    print("- accuracy: 0 (incorrect) or 1 (correct)")
    print()
    
    model, parameters = analyze_csv(file_path)