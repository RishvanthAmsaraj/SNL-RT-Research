# Single-Choice DDM Analysis Script
# This script performs analysis on single-choice Drift Diffusion Model (DDM) data.
# It loads and validates data from a CSV file, estimates DDM parameters,
# simulates decision paths, and generates visualizations to illustrate results.
# The script uses various libraries for data manipulation, statistical analysis,
# optimization, and plotting.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def load_and_validate_data(file_path):
    """
    Loads single-choice DDM data from a CSV file and performs validation checks.

    This function reads the CSV file into a pandas DataFrame, checks for required columns,
    converts relevant columns to numeric types, removes rows with missing values, and validates
    data integrity (e.g., accuracy values, non-negative times, and logical relationships between
    response and decision times). It also filters trials within reasonable time bounds and
    calculates non-decision time for each trial.

    Parameters:
    file_path (str): The path to the CSV file containing the DDM data.

    Returns:
    pd.DataFrame or None: The validated DataFrame if successful, otherwise None if validation fails.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"\nData Loading:")
        print(f"  Dataset loaded: {len(df):,} trials")
        
        # Validate required columns
        expected_cols = ['trial', 'response_time', 'accuracy', 'decision_time', 'evidence_time']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        
        if missing_cols:
            print(f"  Missing columns: {missing_cols}")
            return None
        
        # Convert columns to numeric
        numeric_cols = ['response_time', 'accuracy', 'decision_time', 'evidence_time']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing values
        initial_len = len(df)
        df = df.dropna()
        if len(df) < initial_len:
            print(f"  Removed {initial_len - len(df)} rows with missing data")
        
        # Validate data integrity
        print(f"\nData Validation:")
        
        unique_acc = df['accuracy'].unique()
        if not set(unique_acc).issubset({0, 1}):
            print(f"  Invalid accuracy values: {unique_acc}")
            return None
        
        if df['response_time'].min() <= 0:
            print(f"  Invalid response times found (≤ 0)")
            return None
        
        calculated_ndt = df['response_time'] - df['decision_time']
        negative_ndt = calculated_ndt < 0
        if negative_ndt.any():
            print(f"  Error: {negative_ndt.sum()} trials have response_time < decision_time")
            return None
        
        # Filter trials within reasonable time bounds
        original_len = len(df)
        df = df[
            (df['response_time'] >= 0.1) & (df['response_time'] <= 10.0) &
            (df['decision_time'] >= 0.0) & (df['decision_time'] <= 10.0)
        ]
        
        if len(df) < original_len:
            print(f"  Filtered {original_len - len(df)} trials outside time bounds")
        
        # Calculate non-decision time
        df['calculated_ndt'] = df['response_time'] - df['decision_time']
        
        # Report summary statistics
        ndt_mean = df['calculated_ndt'].mean()
        ndt_std = df['calculated_ndt'].std()
        
        print(f"\nValidation Summary:")
        print(f"  Dataset size: {len(df):,} trials")
        print(f"  Response time range: {df['response_time'].min():.3f}s to {df['response_time'].max():.3f}s")
        print(f"  Decision time range: {df['decision_time'].min():.3f}s to {df['decision_time'].max():.3f}s")
        print(f"  Non-decision time: {ndt_mean:.3f}s ± {ndt_std:.3f}s")
        print(f"  Overall accuracy: {df['accuracy'].mean():.3f}")
        print(f"Data validation completed")
        
        return df
        
    except Exception as e:
        print(f"\nData Loading Error:")
        print(f"  {e}")
        return None

def estimate_ddm_parameters_corrected(df):
    """
    Estimates Drift Diffusion Model (DDM) parameters for single-choice tasks.

    This function first categorizes trials into correct and incorrect, then uses a direct
    estimation method to compute initial DDM parameters (drift rate, boundary, non-decision time,
    and noise). It then attempts a constrained optimization to refine these parameters by minimizing
    a cost function that compares observed and predicted statistics. If optimization fails, it falls
    back to the direct estimates.

    Parameters:
    df (pd.DataFrame): The validated DataFrame containing DDM trial data.

    Returns:
    dict: A dictionary containing the estimated DDM parameters, including 'drift', 'boundary',
          'ndt', 'noise', 'estimation_method', and 'ndt_std'.
    """
    print(f"\nDDM Parameter Estimation:")
    
    # Categorize trials
    correct_trials = df[df['accuracy'] == 1]
    incorrect_trials = df[df['accuracy'] == 0]
    
    print(f"\nTrial Distribution:")
    print(f"  Correct trials: {len(correct_trials)} ({len(correct_trials)/len(df)*100:.1f}%)")
    print(f"  Incorrect trials: {len(incorrect_trials)} ({len(incorrect_trials)/len(df)*100:.1f}%)")
    
    # Direct estimation method
    print(f"\nDirect Estimation:")
    
    ndt_direct = df['calculated_ndt'].mean()
    ndt_std = df['calculated_ndt'].std()
    
    dt_75 = df['decision_time'].quantile(0.75)
    dt_25 = df['decision_time'].quantile(0.25)
    boundary_direct = max(0.5, min(2.0, dt_75 - dt_25 + 0.5))
    
    accuracy = df['accuracy'].mean()
    mean_dt = df['decision_time'].mean()
    
    base_drift = (accuracy - 0.5) * 2.0
    time_factor = max(0.1, boundary_direct / mean_dt)
    drift_direct = max(0.1, min(3.0, base_drift + time_factor))
    
    direct_params = {
        'drift': drift_direct,
        'boundary': boundary_direct,
        'ndt': ndt_direct,
        'noise': 1.0
    }
    
    print(f"  Drift rate: {direct_params['drift']:.3f}")
    print(f"  Boundary: {direct_params['boundary']:.3f}")
    print(f"  Non-decision time: {direct_params['ndt']:.3f}s ± {ndt_std:.3f}s")
    print(f"  Noise: {direct_params['noise']:.3f}")
    
    # Optimization method
    print(f"\nConstrained Optimization:")
    
    def ddm_cost_function(params, data):
        """
        Computes the cost for DDM parameter optimization.

        This inner function calculates a cost value based on the differences between observed
        data statistics (mean decision time, accuracy, standard deviation of decision time) and
        those predicted by the DDM model using the given parameters. It penalizes invalid parameter
        values heavily.

        Parameters:
        params (list): List containing drift, boundary, and ndt values.
        data (pd.DataFrame): The DataFrame with trial data.

        Returns:
        float: The computed cost value (lower is better).
        """
        drift, boundary, ndt = params
        
        if drift <= 0 or boundary <= 0 or ndt <= 0 or ndt >= data['response_time'].min() * 0.8:
            return 1e6
        
        try:
            decision_times = data['response_time'] - ndt
            decision_times = decision_times[decision_times > 0]
            
            if len(decision_times) == 0:
                return 1e6
            
            observed_mean_dt = data['decision_time'].mean()
            observed_accuracy = data['accuracy'].mean()
            observed_std_dt = data['decision_time'].std()
            
            predicted_mean_dt = boundary / drift if drift > 0 else observed_mean_dt * 2
            predicted_accuracy = 1 / (1 + np.exp(-2 * drift * boundary))
            predicted_std_dt = boundary / (drift * np.sqrt(2))
            
            mean_dt_error = ((observed_mean_dt - predicted_mean_dt) / observed_mean_dt) ** 2
            accuracy_error = (observed_accuracy - predicted_accuracy) ** 2
            std_dt_error = ((observed_std_dt - predicted_std_dt) / observed_std_dt) ** 2
            
            total_cost = mean_dt_error + accuracy_error * 2.0 + std_dt_error * 0.5
            
            return total_cost
            
        except:
            return 1e6
    
    initial_guess = [direct_params['drift'], direct_params['boundary'], direct_params['ndt']]
    bounds = [
        (0.1, 3.0),
        (0.3, 2.5),
        (max(0.05, ndt_direct * 0.8), min(ndt_direct * 1.2, df['response_time'].min() * 0.8))
    ]
    
    try:
        result = minimize(ddm_cost_function, initial_guess, args=(df,), 
                         bounds=bounds, method='L-BFGS-B', 
                         options={'maxiter': 1000})
        
        if result.success and result.fun < 100:
            opt_params = {
                'drift': result.x[0],
                'boundary': result.x[1], 
                'ndt': result.x[2],
                'noise': 1.0
            }
            
            print(f"  Optimization completed (cost: {result.fun:.3f})")
            print(f"  Drift rate: {opt_params['drift']:.3f}")
            print(f"  Boundary: {opt_params['boundary']:.3f}")
            print(f"  Non-decision time: {opt_params['ndt']:.3f}s")
            
            final_params = opt_params
            estimation_method = "Optimized"
            
        else:
            print(f"  Optimization failed (cost: {result.fun:.3f})")
            print(f"  Using direct estimation")
            final_params = direct_params
            estimation_method = "Direct"
            
    except Exception as e:
        print(f"  Optimization error: {e}")
        print(f"  Using direct estimation")
        final_params = direct_params
        estimation_method = "Direct"
    
    final_params['estimation_method'] = estimation_method
    final_params['ndt_std'] = ndt_std
    
    print(f"\nFinal Parameters ({estimation_method} method):")
    for key, value in final_params.items():
        if key not in ['estimation_method', 'ndt_std']:
            print(f"  {key.capitalize()}: {value:.3f}")
    
    return final_params

def simulate_ddm_paths(drift, boundary, ndt, n_trials=20, dt=0.001, max_time=3.0):
    """
    Simulates evidence accumulation paths for a single-choice Drift Diffusion Model (DDM).

    This function generates multiple simulated decision paths based on the provided DDM parameters.
    Each path represents the accumulation of evidence over time until a decision boundary is reached
    or the maximum time is exceeded. The simulation uses a discrete-time approximation with random
    noise to mimic the stochastic nature of decision-making.

    Parameters:
    drift (float): The drift rate parameter (average evidence accumulation per unit time).
    boundary (float): The decision boundary threshold.
    ndt (float): The non-decision time (added to the decision time to get response time).
    n_trials (int, optional): Number of paths to simulate. Defaults to 20.
    dt (float, optional): Time step for simulation. Defaults to 0.001 seconds.
    max_time (float, optional): Maximum simulation time per path. Defaults to 3.0 seconds.

    Returns:
    list: A list of dictionaries, each containing 'time', 'evidence', 'decided', and 'rt' for a simulated path.
    """
    paths = []
    
    for _ in range(n_trials):
        evidence = [0.0]
        time_points = [0.0]
        current_time = 0.0
        
        while current_time < max_time:
            current_evidence = evidence[-1] + drift * dt + np.random.normal(0, np.sqrt(dt))
            current_time += dt
            
            evidence.append(current_evidence)
            time_points.append(current_time)
            
            if current_evidence >= boundary:
                break
        
        paths.append({
            'time': time_points,
            'evidence': evidence,
            'decided': evidence[-1] >= boundary,
            'rt': time_points[-1] + ndt
        })
    
    return paths

def create_main_analysis_figure(df, params):
    """
    Generates the main analysis figure for single-choice DDM results.

    This function creates a multi-panel figure (2x3 grid) visualizing various aspects of the data
    and model, including distributions of response and decision times, relationships between times,
    estimated parameters, model fit, and a text summary of key statistics.

    Parameters:
    df (pd.DataFrame): The validated DataFrame with trial data.
    params (dict): Dictionary of estimated DDM parameters.

    Returns:
    None: Displays the figure using matplotlib.
    """
    print(f"\nVisualization:")
    print(f"  Creating main analysis figure")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Single-Choice DDM Analysis: Main Results', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    ax.hist(df['response_time'], bins=30, alpha=0.7, color='skyblue', 
           density=True, edgecolor='black')
    ax.axvline(df['response_time'].mean(), color='red', linestyle='--', 
              linewidth=2, label=f'Mean: {df["response_time"].mean():.3f}s')
    ax.axvline(df['response_time'].median(), color='green', linestyle='--', 
              linewidth=2, label=f'Median: {df["response_time"].median():.3f}s')
    ax.set_xlabel('Response Time (s)')
    ax.set_ylabel('Density')
    ax.set_title('Response Time Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    correct_data = df[df['accuracy'] == 1]
    incorrect_data = df[df['accuracy'] == 0]
    
    ax.hist(df['decision_time'], bins=25, alpha=0.5, color='lightblue', 
           density=True, edgecolor='black', label=f'All trials (n={len(df)})')
    if len(correct_data) > 0:
        ax.hist(correct_data['decision_time'], bins=20, alpha=0.7, color='green', 
               density=True, label=f'Correct (n={len(correct_data)})')
    if len(incorrect_data) > 0:
        ax.hist(incorrect_data['decision_time'], bins=15, alpha=0.7, color='red', 
               density=True, label=f'Incorrect (n={len(incorrect_data)})')
    
    ax.set_xlabel('Decision Time (s)')
    ax.set_ylabel('Density')
    ax.set_title('Decision Time Distribution by Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    scatter = ax.scatter(df['decision_time'], df['response_time'], 
                        alpha=0.6, s=15, c=df['accuracy'], 
                        cmap='RdYlGn', edgecolors='none')
    
    dt_range = [df['decision_time'].min(), df['decision_time'].max()]
    rt_expected = [dt + params['ndt'] for dt in dt_range]
    ax.plot(dt_range, rt_expected, 'r--', linewidth=2, 
           label=f'Expected (NDT={params["ndt"]:.3f}s)')
    
    ax.set_xlabel('Decision Time (s)')
    ax.set_ylabel('Response Time (s)')
    ax.set_title('Response Time vs Decision Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Accuracy')
    
    ax = axes[1, 0]
    param_names = ['Drift\nRate', 'Boundary', 'Non-Decision\nTime']
    param_values = [params['drift'], params['boundary'], params['ndt']]
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    
    bars = ax.bar(param_names, param_values, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1)
    ax.set_ylabel('Parameter Value')
    ax.set_title(f'Estimated DDM Parameters\n({params["estimation_method"]} Method)')
    ax.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, param_values):
        ax.text(bar.get_x() + bar.get_width()/2, 
               bar.get_height() + max(param_values) * 0.02, 
               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax = axes[1, 1]
    sim_paths = simulate_ddm_paths(params['drift'], params['boundary'], 
                                  params['ndt'], n_trials=len(df))
    sim_rts = [path['rt'] for path in sim_paths if path['decided']]
    
    if sim_rts:
        ax.hist(df['response_time'], bins=25, alpha=0.6, density=True, 
               color='blue', label='Observed Data')
        ax.hist(sim_rts, bins=25, alpha=0.6, density=True, 
               color='red', label='Simulated DDM')
        
        obs_mean = df['response_time'].mean()
        sim_mean = np.mean(sim_rts)
        mean_diff = abs(obs_mean - sim_mean)
        
        ax.set_xlabel('Response Time (s)')
        ax.set_ylabel('Density')
        ax.set_title(f'Model Fit Assessment\nMean Difference: {mean_diff:.3f}s')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Simulation\nFailed', transform=ax.transAxes, 
               ha='center', va='center', fontsize=16)
        ax.set_title('Model Fit Assessment')
    
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    ax.axis('off')
    
    n_trials = len(df)
    n_correct = len(df[df['accuracy'] == 1])
    accuracy = df['accuracy'].mean()
    mean_rt = df['response_time'].mean()
    mean_dt = df['decision_time'].mean()
    mean_ndt = df['calculated_ndt'].mean()
    std_ndt = df['calculated_ndt'].std()
    
    if len(correct_data) > 10 and len(incorrect_data) > 10:
        t_stat, p_val = stats.ttest_ind(correct_data['response_time'], 
                                       incorrect_data['response_time'])
        rt_diff_text = f't={t_stat:.2f}, p={p_val:.4f}'
    else:
        rt_diff_text = 'N/A (insufficient data)'
    
    summary_text = f"""ANALYSIS SUMMARY

Dataset:
• Total trials: {n_trials:,}
• Accuracy: {accuracy:.1%} ({n_correct}/{n_trials})

Timing (seconds):
• Mean RT: {mean_rt:.3f}
• Mean Decision Time: {mean_dt:.3f}
• Mean NDT: {mean_ndt:.3f} ± {std_ndt:.3f}

DDM Parameters:
• Drift Rate: {params['drift']:.3f}
• Boundary: {params['boundary']:.3f}
• Non-Decision Time: {params['ndt']:.3f}
• Estimation: {params['estimation_method']}

Model Quality:
• NDT Consistency: {std_ndt:.4f}
• RT Difference Test: {rt_diff_text}

Data Quality:
• No negative NDTs
• Reasonable parameter values
• Consistent timing relationships
    """
    
    ax.text(0.05, 0.95, summary_text.strip(), transform=ax.transAxes,
           verticalalignment='top', fontsize=10, fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.show(block=False)

def create_process_visualization(params):
    """
    Generates a visualization of the DDM process under varying parameters.

    This function creates a multi-panel figure (2x3 grid) showing simulated decision paths
    for variations in drift rate (top row) and boundary (bottom row). Each panel illustrates
    how changes in these parameters affect evidence accumulation and decision-making.

    Parameters:
    params (dict): Dictionary of estimated DDM parameters.

    Returns:
    None: Displays the figure using matplotlib.
    """
    print(f"  Creating process visualization")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('DDM Process Visualization: Parameter Effects', fontsize=16, fontweight='bold')
    
    drift_variations = [params['drift'] * 0.6, params['drift'], params['drift'] * 1.4]
    boundary_variations = [params['boundary'] * 0.7, params['boundary'], params['boundary'] * 1.3]
    
    for i, drift in enumerate(drift_variations):
        ax = axes[0, i]
        paths = simulate_ddm_paths(drift, params['boundary'], params['ndt'], n_trials=15)
        
        ax.axhline(params['boundary'], color='red', linewidth=3, alpha=0.8,
                  label=f'Boundary ({params["boundary"]:.2f})')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        decided_count = 0
        for path in paths:
            color = 'green' if path['decided'] else 'gray'
            alpha = 0.8 if path['decided'] else 0.4
            ax.plot(path['time'], path['evidence'], color=color, alpha=alpha, linewidth=1)
            if path['decided']:
                decided_count += 1
        
        multiplier = drift / params['drift']
        ax.set_xlim(0, 2.5)
        ax.set_ylim(-0.3, max(boundary_variations) * 1.2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Evidence')
        ax.set_title(f'Drift Rate × {multiplier:.1f}\n(v = {drift:.2f})\nDecisions: {decided_count}/15')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend()
    
    for i, boundary in enumerate(boundary_variations):
        ax = axes[1, i]
        paths = simulate_ddm_paths(params['drift'], boundary, params['ndt'], n_trials=15)
        
        ax.axhline(boundary, color='red', linewidth=3, alpha=0.8,
                  label=f'Boundary ({boundary:.2f})')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        decided_count = 0
        for path in paths:
            color = 'green' if path['decided'] else 'gray'
            alpha = 0.8 if path['decided'] else 0.4
            ax.plot(path['time'], path['evidence'], color=color, alpha=alpha, linewidth=1)
            if path['decided']:
                decided_count += 1
        
        multiplier = boundary / params['boundary']
        ax.set_xlim(0, 2.5)
        ax.set_ylim(-0.3, max(boundary_variations) * 1.2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Evidence')
        ax.set_title(f'Boundary × {multiplier:.1f}\n(a = {boundary:.2f})\nDecisions: {decided_count}/15')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.show(block=False)

def run_corrected_analysis(file_path):
    """
    Executes the complete single-choice DDM analysis pipeline.

    This function orchestrates the entire analysis process: loading and validating data,
    estimating parameters, generating the main analysis figure, creating process visualizations,
    and printing a final summary. It handles errors by returning early if data issues are detected.

    Parameters:
    file_path (str): The path to the CSV file containing the DDM data.

    Returns:
    tuple or None: A tuple (df, params) if successful, otherwise None.
    """
    print(f"\nAnalysis Execution:")
    print(f"  Single-choice DDM analysis started")
    
    df = load_and_validate_data(file_path)
    if df is None:
        print(f"  Analysis failed due to data issues")
        return None
    
    params = estimate_ddm_parameters_corrected(df)
    
    create_main_analysis_figure(df, params)
    create_process_visualization(params)
    
    print(f"\nAnalysis Summary:")
    print(f"  Single-choice DDM analysis completed")
    print(f"  Dataset: {len(df):,} trials, {df['accuracy'].mean():.1%} accuracy")
    print(f"  Drift rate: {params['drift']:.3f}")
    print(f"  Boundary: {params['boundary']:.3f}")
    print(f"  Non-decision time: {params['ndt']:.3f}s")
    print(f"  Estimation method: {params['estimation_method']}")
    
    input("Press Enter to close all figures...")
    plt.close('all')
    
    return df, params

# Main execution
if __name__ == "__main__":
    file_path = r"C:\Users\Rishv\OneDrive\Documents\Personal\Projects\PyDDM\Single Choice\Data\single_choice_ddm_extended.csv"
    results = run_corrected_analysis(file_path)