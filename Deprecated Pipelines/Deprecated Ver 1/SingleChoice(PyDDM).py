# Simplified PyDDM Single Choice CSV Analysis with Comprehensive Visualizations
# This script performs analysis on single-choice Drift Diffusion Model (DDM) data using PyDDM.
# It loads and prepares data from a CSV file, fits a DDM model, simulates trajectories,
# and generates comprehensive visualizations to illustrate the results.
# The script handles both response_time and optional decision_time columns.

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

def load_csv_for_single_choice_pyddm(file_path):
    """
    Load and prepare CSV data for single-choice PyDDM analysis.

    This function reads the CSV file into a pandas DataFrame, converts relevant columns to numeric types,
    cleans the data by removing invalid trials, and selects the appropriate time column for analysis
    (preferring 'decision_time' if available, otherwise 'response_time').

    Parameters:
    file_path (str): The path to the CSV file containing the single-choice DDM data.

    Returns:
    tuple: A tuple containing the cleaned DataFrame and the name of the time column used for analysis.
    """
    # Load data
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} trials")
    print(f"Columns found: {list(df.columns)}")
    
    # Convert to numeric and clean data
    df['response_time'] = pd.to_numeric(df['response_time'], errors='coerce')
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    
    # For single choice, we can also use decision_time if available
    if 'decision_time' in df.columns:
        df['decision_time'] = pd.to_numeric(df['decision_time'], errors='coerce')
        print("Using decision_time for analysis")
        time_column = 'decision_time'
    else:
        time_column = 'response_time'
    
    # Remove invalid trials
    df = df.dropna(subset=[time_column, 'accuracy'])
    df = df[(df[time_column] > 0) & (df[time_column] < 10)]
    df = df[df['accuracy'].isin([0, 1])]
    
    print(f"After cleaning: {len(df)} trials")
    print(f"Accuracy: {df['accuracy'].mean():.3f}")
    print(f"Mean {time_column}: {df[time_column].mean():.3f}s")
    
    return df, time_column

def fit_single_choice_pyddm_model(df, time_column):
    """
    Fit a single-choice PyDDM model to the provided data.

    This function prepares a PyDDM Sample from the DataFrame, defines a DDM model suitable for
    single-choice tasks (modeling correct vs. incorrect responses), and fits the model using either
    robust or standard methods. It handles warnings and errors during the fitting process.

    Parameters:
    df (pd.DataFrame): The cleaned DataFrame containing trial data.
    time_column (str): The name of the column to use for response/decision times.

    Returns:
    tuple: The fitted PyDDM Model (or None if fitting fails) and a dictionary of fitted parameters.
    """
    # For single choice, we model the decision as "respond" vs "no response"
    # or "correct" vs "incorrect" response
    df_sample = df.copy()
    
    # Create choice variable - in single choice, this could be:
    # 1) Accuracy-based: correct (1) vs incorrect (0)
    # 2) Response-based: responded (1) vs timeout (0)
    df_sample['choice'] = df_sample['accuracy']  # Using accuracy as choice
    df_sample['rt'] = df_sample[time_column]
    
    # Create PyDDM sample for single choice
    try:
        sample = Sample.from_pandas_dataframe(
            df_sample,
            rt_column_name="rt",
            choice_column_name="choice",
            choice_names=("incorrect", "correct")  # Still need two boundaries for PyDDM
        )
        print(f"Created single choice PyDDM sample with {len(sample)} trials")
    except Exception as e:
        print(f"Error creating sample: {e}")
        return None, {}
    
    # Define single choice model
    # For single choice, we often use a simpler model
    model = Model(
        name="Single Choice DDM",
        drift=DriftConstant(drift=Fittable(minval=0, maxval=5, value=1.5)),  # Positive drift for single choice
        noise=NoiseConstant(noise=1),  # Fixed noise
        bound=BoundConstant(B=Fittable(minval=0.5, maxval=4, value=1.5)),
        overlay=OverlayChain(overlays=[
            OverlayNonDecision(nondectime=Fittable(minval=0, maxval=1.0, value=0.25)),
            OverlayPoissonMixture(
                pmixturecoef=Fittable(minval=0.0, maxval=0.3, value=0.05),
                rate=Fittable(minval=0.1, maxval=3.0, value=1.0)
            )
        ]),
        dx=0.01, dt=0.01, T_dur=6.0,
        choice_names=("incorrect", "correct")
    )
    
    # Fit model with error suppression
    try:
        print("Fitting single choice model...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if ROBUST_AVAILABLE:
                model.fit(sample, lossfunction=LossRobustBIC, verbose=False)
                print("Single choice model fitted successfully (robust method)!")
            else:
                model.fit(sample, verbose=False)
                print("Single choice model fitted successfully (standard method)!")
        
        # Extract parameters
        param_names = model.get_model_parameter_names()
        param_values = model.get_model_parameters()
        fitted_params = dict(zip(param_names, param_values))
        
        print("\nFitted Parameters:")
        for name, value in fitted_params.items():
            print(f"  {name}: {value:.4f}")
            
        return model, fitted_params
        
    except Exception as e:
        print(f"Error fitting single choice model: {e}")
        return None, {}

def simulate_single_choice_trajectories(drift, boundary, ndt, noise=1.0, n_trajectories=50, dt=0.01, max_time=4.0):
    """
    Simulate evidence accumulation trajectories for a single-choice DDM.

    This function generates multiple simulated trajectories based on the provided DDM parameters.
    Each trajectory accumulates evidence over time with drift and noise until it hits a decision boundary,
    an error/timeout boundary, or the maximum time is reached.

    Parameters:
    drift (float): The drift rate parameter.
    boundary (float): The decision boundary threshold.
    ndt (float): The non-decision time.
    noise (float, optional): The noise level. Defaults to 1.0.
    n_trajectories (int, optional): Number of trajectories to simulate. Defaults to 50.
    dt (float, optional): Time step for simulation. Defaults to 0.01 seconds.
    max_time (float, optional): Maximum simulation time per trajectory. Defaults to 4.0 seconds.

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
            # Update evidence (positive drift for single choice)
            dW = np.random.randn() * np.sqrt(dt)
            x += drift * dt + noise * dW
            t += dt
            
            time_points.append(t)
            evidence.append(x)
            
            # Check boundary (single choice typically has one decision boundary)
            if x >= boundary:
                choice = 1  # decision made
                decided = True
            # In single choice, we might also have a lower boundary for "no response" or errors
            elif x <= -boundary * 0.5:  # Lower threshold for errors/timeouts
                choice = 0  # no decision or error
                decided = True
        
        if not decided:
            choice = 1 if x > 0 else 0  # forced choice at timeout
        
        rt = t + ndt
        
        trajectories.append({
            'time': time_points,
            'evidence': evidence,
            'choice': choice,
            'rt': rt,
            'decided': decided
        })
    
    return trajectories

def plot_single_choice_results(df, model, params, time_column):
    """
    Create comprehensive plots for single-choice DDM analysis results.

    This function generates a 2x3 grid of plots visualizing response time distributions,
    performance metrics, parameter values, simulated trajectories, and a summary text box.
    It handles cases where model fitting failed by providing basic visualizations.

    Parameters:
    df (pd.DataFrame): The cleaned DataFrame with trial data.
    model (pyddm.Model or None): The fitted PyDDM model.
    params (dict): Dictionary of fitted parameters.
    time_column (str): The name of the time column used.

    Returns:
    None: Displays the figure using matplotlib.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Single Choice PyDDM Analysis Results', fontsize=16, fontweight='bold')
    
    # Plot 1: RT distributions
    ax = axes[0, 0]
    correct_trials = df[df['accuracy'] == 1][time_column]
    incorrect_trials = df[df['accuracy'] == 0][time_column]
    
    ax.hist(df[time_column], bins=30, alpha=0.5, color='lightblue', 
           density=True, label=f'All trials (n={len(df)})')
    if len(correct_trials) > 5:
        ax.hist(correct_trials, bins=25, alpha=0.7, color='green', 
               density=True, label=f'Correct (n={len(correct_trials)})')
    if len(incorrect_trials) > 5:
        ax.hist(incorrect_trials, bins=25, alpha=0.7, color='red', 
               density=True, label=f'Incorrect (n={len(incorrect_trials)})')
    
    ax.set_xlabel(f'{time_column.replace("_", " ").title()} (s)')
    ax.set_ylabel('Density')
    ax.set_title('Response Time Distributions (Single Choice)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy and Response Rate
    ax = axes[0, 1]
    accuracy = df['accuracy'].mean()
    response_rate = len(df[df[time_column] < df[time_column].quantile(0.95)]) / len(df)
    
    ax.bar(['Accuracy', 'Response Rate'], [accuracy, response_rate], 
           color=['skyblue', 'lightcoral'], alpha=0.7)
    ax.set_ylabel('Proportion')
    ax.set_title('Single Choice Performance')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: RT by Accuracy with additional metrics
    ax = axes[0, 2]
    if len(correct_trials) > 0 and len(incorrect_trials) > 0:
        correct_rt = correct_trials.mean()
        incorrect_rt = incorrect_trials.mean()
        ax.bar(['Correct', 'Incorrect'], [correct_rt, incorrect_rt], 
               color=['green', 'red'], alpha=0.7)
        ax.set_ylabel(f'Mean {time_column.replace("_", " ").title()} (s)')
        ax.set_title('Response Time by Accuracy')
    else:
        ax.bar(['Overall'], [df[time_column].mean()], color='skyblue', alpha=0.7)
        ax.set_ylabel(f'Mean {time_column.replace("_", " ").title()} (s)')
        ax.set_title('Overall Response Time')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Parameter values
    ax = axes[1, 0]
    param_names = ['Drift Rate', 'Boundary', 'Non-Dec Time', 'Mix Coef']
    param_keys = ['drift', 'B', 'nondectime', 'pmixturecoef']
    param_values = [params.get(key, 0) for key in param_keys]
    colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
    
    bars = ax.bar(param_names, param_values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Parameter Value')
    ax.set_title('Fitted Single Choice DDM Parameters')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, param_values):
        if not np.isnan(value) and value != 0:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, 
                       height + max([v for v in param_values if v > 0]) * 0.02, 
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: Single Choice Model Trajectories
    ax = axes[1, 1]
    if params:
        drift = params.get('drift', 1.0)
        boundary = params.get('B', 1.5)
        ndt = params.get('nondectime', 0.25)
        
        trajectories = simulate_single_choice_trajectories(drift, boundary, ndt, n_trajectories=30)
        
        # Single choice boundary visualization
        ax.axhline(boundary, color='green', linewidth=3, alpha=0.8, label='Decision Boundary')
        ax.axhline(-boundary * 0.5, color='red', linewidth=2, alpha=0.6, label='Error/Timeout Boundary')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Starting Point')
        
        for traj in trajectories:
            color = 'green' if traj['choice'] == 1 else 'red'
            alpha = 0.6 if traj['decided'] else 0.3
            linewidth = 1.5 if traj['choice'] == 1 else 1
            ax.plot(traj['time'], traj['evidence'], color=color, alpha=alpha, linewidth=linewidth)
        
        ax.set_xlim(0, 3)
        ax.set_ylim(-boundary, boundary * 1.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Evidence Accumulation')
        ax.set_title('Single Choice DDM Trajectories')
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
    mean_time = df[time_column].mean()
    median_time = df[time_column].median()
    
    correct_mean_time = df[df['accuracy'] == 1][time_column].mean() if len(df[df['accuracy'] == 1]) > 0 else np.nan
    incorrect_mean_time = df[df['accuracy'] == 0][time_column].mean() if len(df[df['accuracy'] == 0]) > 0 else np.nan
    
    summary_text = f"""SINGLE CHOICE ANALYSIS

Dataset:
• Total trials: {n_trials:,}
• Overall accuracy: {overall_accuracy:.3f}
• Success rate: {(df['accuracy'] == 1).sum()}/{n_trials}

Response Times:
• Mean time: {mean_time:.3f}s
• Median time: {median_time:.3f}s"""

    if not np.isnan(correct_mean_time):
        summary_text += f"\n• Correct time: {correct_mean_time:.3f}s"
    if not np.isnan(incorrect_mean_time):
        summary_text += f"\n• Incorrect time: {incorrect_mean_time:.3f}s"

    summary_text += f"""

Model Parameters:
• Drift rate: {params.get('drift', 'N/A'):.3f}
• Boundary: {params.get('B', 'N/A'):.3f}
• Non-decision: {params.get('nondectime', 'N/A'):.3f}s
• Mixture coef: {params.get('pmixturecoef', 'N/A'):.3f}

Model Quality:
• Fitting: {'Success' if params else 'Failed'}
• Method: {'Robust' if ROBUST_AVAILABLE else 'Standard'}
• Model type: Single Choice DDM"""
    
    ax.text(0.05, 0.95, summary_text.strip(), transform=ax.transAxes,
           verticalalignment='top', fontsize=10, fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def analyze_single_choice_csv(file_path):
    """
    Main function to perform single-choice PyDDM analysis on a CSV file.

    This function orchestrates the entire analysis: loading data, fitting the model,
    and generating visualizations. It provides console output for progress and results,
    and handles cases where model fitting fails by still producing basic plots.

    Parameters:
    file_path (str): The path to the CSV file containing the data.

    Returns:
    tuple: The fitted model (or None) and a dictionary of parameters.
    """
    print("=" * 60)
    print("Single Choice PyDDM Analysis")
    print("=" * 60)
    
    # Load data
    df, time_column = load_csv_for_single_choice_pyddm(file_path)
    
    # Fit model
    model, params = fit_single_choice_pyddm_model(df, time_column)
    
    if model is not None:
        print("\n" + "=" * 60)
        print("Single Choice Analysis Complete!")
        print("=" * 60)
        
        # Create comprehensive plots
        plot_single_choice_results(df, model, params, time_column)
        
        return model, params
    else:
        print("Single choice analysis failed.")
        # Still create basic plots even if model fitting failed
        print("Creating basic visualizations...")
        plot_single_choice_results(df, None, {}, time_column)
        return None, {}

# Usage example
if __name__ == "__main__":
    file_path = r"C:\Users\Rishv\OneDrive\Documents\Personal\Projects\PyDDM\Single Choice\Data\single_choice_ddm_extended.csv"
    
    print("Expected CSV format for Single Choice:")
    print("Required columns:")
    print("- response_time: reaction times in seconds")
    print("- accuracy: 0 (incorrect/timeout) or 1 (correct/responded)")
    print("Optional columns:")
    print("- decision_time: decision times (if different from response_time)")
    print("- evidence_time: evidence accumulation time")
    print()
    
    model, parameters = analyze_single_choice_csv(file_path)