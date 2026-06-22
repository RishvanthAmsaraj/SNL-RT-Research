# PyDDM CSV Analysis Tool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

try:
    import pyddm
    print(f"PyDDM base module imported successfully")
    
    from pyddm import Model, Fittable, Sample
    print("Core PyDDM classes imported")
    
    from pyddm.models import Drift, NoiseConstant, BoundConstant
    print("PyDDM model components imported")
    
    from pyddm.models import OverlayChain, OverlayNonDecision, OverlayPoissonMixture
    print("PyDDM overlay components imported")
    
    try:
        from pyddm.functions import LossRobustBIC, LossRobustLikelihood
        print("PyDDM robust loss functions imported")
        ROBUST_LOSS_AVAILABLE = True
    except ImportError:
        print("Robust loss functions not available, will use default")
        ROBUST_LOSS_AVAILABLE = False
    
    PYDDM_AVAILABLE = True
    print("✓ PyDDM successfully imported and ready to use")
    
except ImportError as e:
    PYDDM_AVAILABLE = False
    ROBUST_LOSS_AVAILABLE = False
    print(f"✗ PyDDM not available: {e}")
    print("Will use built-in DDM analysis instead")
    print("To install PyDDM, try: pip install pyddm")
except Exception as e:
    PYDDM_AVAILABLE = False
    ROBUST_LOSS_AVAILABLE = False
    print(f"✗ Error importing PyDDM: {e}")
    print("Will use built-in DDM analysis instead")

def load_and_validate_csv_data(file_path):
    """Loads and validates DDM data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"\nData Loading:")
        print(f"  Dataset loaded: {len(df):,} trials")
        
        # Check for required columns
        required_cols = ['response_time', 'accuracy']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"  Missing required columns: {missing_cols}")
            print(f"  Available columns: {list(df.columns)}")
            return None
        
        # Convert to numeric
        df['response_time'] = pd.to_numeric(df['response_time'], errors='coerce')
        df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
        
        # Remove invalid data
        initial_len = len(df)
        df = df.dropna(subset=['response_time', 'accuracy'])
        df = df[(df['response_time'] > 0) & (df['response_time'] < 10)]
        df = df[df['accuracy'].isin([0, 1])]
        
        if len(df) < initial_len:
            print(f"  Removed {initial_len - len(df)} invalid trials")
        
        # Check for condition column (optional)
        if 'condition' in df.columns:
            df['condition'] = pd.to_numeric(df['condition'], errors='coerce')
            print(f"  Found condition column with {df['condition'].nunique()} unique values")
        else:
            # Create default condition
            df['condition'] = 1.0
            print(f"  No condition column found, using default condition = 1.0")
        
        print(f"\nData Summary:")
        print(f"  Final dataset: {len(df):,} trials")
        print(f"  Response time range: {df['response_time'].min():.3f}s to {df['response_time'].max():.3f}s")
        print(f"  Overall accuracy: {df['accuracy'].mean():.3f}")
        print(f"  Conditions: {sorted(df['condition'].unique())}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

class DriftCoherence(Drift):
    """Drift rate that depends on condition/coherence."""
    name = "Drift depending on condition"
    required_parameters = ["drift_base", "drift_scale"]
    required_conditions = ["condition"]

    def get_drift(self, x, t, conditions, **kwargs):
        condition = conditions.get("condition", 1.0)
        return self.drift_base + self.drift_scale * condition

def prepare_pyddm_sample(df):
    """Converts DataFrame to PyDDM Sample format."""
    # Create choice column: 1 for correct, 0 for incorrect
    df_sample = df.copy()
    df_sample['choice'] = df_sample['accuracy']
    
    print(f"  Converting DataFrame to PyDDM Sample...")
    
    # Try different parameter names for different PyDDM versions
    try:
        # Try the newer parameter name format
        sample = Sample.from_pandas_dataframe(
            df_sample,
            rt_column_name="response_time",
            choice_column_name="choice", 
            choice_names=("incorrect", "correct"),
            condition_column_names=["condition"]
        )
        print(f"  ✓ PyDDM Sample created (method 1): {len(sample)} trials")
        return sample
    except TypeError as e:
        print(f"  Method 1 failed: {e}")
        
    try:
        # Try alternative parameter name
        sample = Sample.from_pandas_dataframe(
            df_sample,
            "response_time",
            "choice",
            choice_names=("incorrect", "correct"),
            conditions=["condition"]
        )
        print(f"  ✓ PyDDM Sample created (method 2): {len(sample)} trials")
        return sample
    except TypeError as e:
        print(f"  Method 2 failed: {e}")
        
    try:
        # Try with positional arguments
        sample = Sample.from_pandas_dataframe(
            df_sample,
            "response_time", 
            "choice",
            ("incorrect", "correct"),
            "condition"
        )
        print(f"  ✓ PyDDM Sample created (method 3): {len(sample)} trials")
        return sample
    except TypeError as e:
        print(f"  Method 3 failed: {e}")
        
    try:
        # Try without condition column
        sample = Sample.from_pandas_dataframe(
            df_sample,
            "response_time",
            "choice", 
            ("incorrect", "correct")
        )
        print(f"  ✓ PyDDM Sample created (method 4, no conditions): {len(sample)} trials")
        return sample
    except Exception as e:
        print(f"  Method 4 failed: {e}")
        
    # If all methods fail, return None
    print(f"  ✗ All methods failed to create PyDDM Sample")
    return None

def create_pyddm_model():
    """Creates a PyDDM model for fitting."""
    model = Model(
        name="DDM with Condition-dependent Drift",
        drift=DriftCoherence(
            drift_base=Fittable(minval=-3.0, maxval=3.0, value=0.5),
            drift_scale=Fittable(minval=0.0, maxval=5.0, value=1.0)
        ),
        noise=NoiseConstant(noise=1.0),  # Fixed noise
        bound=BoundConstant(B=Fittable(minval=0.3, maxval=3.0, value=1.0)),
        overlay=OverlayChain(overlays=[
            OverlayNonDecision(nondectime=Fittable(minval=0.0, maxval=0.8, value=0.2)),
            OverlayPoissonMixture(
                pmixturecoef=Fittable(minval=0.0, maxval=0.2, value=0.05),
                rate=Fittable(minval=0.1, maxval=2.0, value=1.0)
            )
        ]),
        dx=0.01, dt=0.01, T_dur=5.0,
        choice_names=("incorrect", "correct")
    )
    
    return model

def fit_pyddm_model(sample):
    """Fits PyDDM model to data."""
    print(f"\nModel Fitting:")
    print(f"  Creating DDM model")
    
    model = create_pyddm_model()
    
    print(f"  Fitting model to data...")
    fitting_successful = False
    
    if ROBUST_LOSS_AVAILABLE:
        try:
            # Try robust BIC first
            model.fit(sample, lossfunction=LossRobustBIC, verbose=False)
            print(f"  ✓ Model fitted successfully (Robust BIC)")
            fitting_successful = True
        except Exception as e:
            print(f"  ✗ Robust BIC failed: {e}")
            try:
                # Fall back to robust likelihood
                model.fit(sample, lossfunction=LossRobustLikelihood, verbose=False)
                print(f"  ✓ Model fitted successfully (Robust Likelihood)")
                fitting_successful = True
            except Exception as e:
                print(f"  ✗ Robust Likelihood failed: {e}")
    
    if not fitting_successful:
        try:
            # Fall back to default
            model.fit(sample, verbose=False)
            print(f"  ✓ Model fitted successfully (Default method)")
            fitting_successful = True
        except Exception as e:
            print(f"  ✗ All fitting methods failed: {e}")
            print(f"  This might be due to data issues or PyDDM version compatibility")
    
    if not fitting_successful:
        return None, {}
    
    # Extract fitted parameters
    try:
        param_names = model.get_model_parameter_names()
        param_values = model.get_model_parameters()
        fitted_params = dict(zip(param_names, param_values))
        
        print(f"\nFitted Parameters:")
        for name, value in fitted_params.items():
            print(f"  {name}: {value:.4f}")
            
    except Exception as e:
        print(f"  Error extracting parameters: {e}")
        fitted_params = {}
    
    return model, fitted_params

def generate_model_predictions(model, df):
    """Generates model predictions for comparison."""
    print(f"\nGenerating Model Predictions:")
    
    conditions = sorted(df['condition'].unique())
    predictions = {}
    
    for condition in conditions:
        try:
            # Create sample for this condition
            condition_data = df[df['condition'] == condition].copy()
            condition_data['choice'] = condition_data['accuracy']
            
            # Use the same flexible approach as in prepare_pyddm_sample
            condition_sample = None
            
            # Try different methods to create condition sample
            try:
                condition_sample = Sample.from_pandas_dataframe(
                    condition_data,
                    "response_time", 
                    "choice",
                    ("incorrect", "correct"),
                    "condition"
                )
            except:
                try:
                    condition_sample = Sample.from_pandas_dataframe(
                        condition_data,
                        "response_time",
                        "choice", 
                        ("incorrect", "correct")
                    )
                except:
                    print(f"  Warning: Could not create sample for condition {condition}")
                    continue
            
            if condition_sample is None:
                continue
            
            # Generate model solution
            solution = model.solve(conditions={"condition": condition})
            
            # Extract predictions
            pred_correct_prob = solution.prob("correct")
            pred_correct_rt = solution.mean_decision_time("correct")
            pred_incorrect_rt = solution.mean_decision_time("incorrect")
            
            # Observed data
            obs_correct_prob = condition_data['accuracy'].mean()
            obs_correct_rt = condition_data[condition_data['accuracy'] == 1]['response_time'].mean()
            obs_incorrect_rt = condition_data[condition_data['accuracy'] == 0]['response_time'].mean()
            
            predictions[condition] = {
                'pred_correct_prob': pred_correct_prob,
                'pred_correct_rt': pred_correct_rt,
                'pred_incorrect_rt': pred_incorrect_rt,
                'obs_correct_prob': obs_correct_prob,
                'obs_correct_rt': obs_correct_rt,
                'obs_incorrect_rt': obs_incorrect_rt,
                'n_trials': len(condition_data)
            }
            
            print(f"  Condition {condition}: {pred_correct_prob:.3f} acc, {pred_correct_rt:.3f}s RT")
            
        except Exception as e:
            print(f"  Error predicting condition {condition}: {e}")
            
            # Fall back to observed data only
            condition_data = df[df['condition'] == condition]
            predictions[condition] = {
                'pred_correct_prob': None,
                'pred_correct_rt': None,
                'pred_incorrect_rt': None,
                'obs_correct_prob': condition_data['accuracy'].mean(),
                'obs_correct_rt': condition_data[condition_data['accuracy'] == 1]['response_time'].mean(),
                'obs_incorrect_rt': condition_data[condition_data['accuracy'] == 0]['response_time'].mean(),
                'n_trials': len(condition_data)
            }
    
    return predictions

def simulate_ddm_trajectories(drift, boundary, ndt, noise=1.0, n_trajectories=50, dt=0.01, max_time=3.0):
    """Simulates DDM trajectories for visualization."""
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

def create_comprehensive_visualization(df, model, fitted_params, predictions):
    """Creates comprehensive visualization of DDM analysis results."""
    print(f"\nCreating Visualizations:")
    
    # Main analysis figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('PyDDM Analysis Results', fontsize=16, fontweight='bold')
    
    # 1. Response Time Distribution
    ax = axes[0, 0]
    correct_trials = df[df['accuracy'] == 1]
    incorrect_trials = df[df['accuracy'] == 0]
    
    ax.hist(df['response_time'], bins=30, alpha=0.5, color='lightblue', 
           density=True, label=f'All trials (n={len(df)})')
    if len(correct_trials) > 10:
        ax.hist(correct_trials['response_time'], bins=25, alpha=0.7, color='green', 
               density=True, label=f'Correct (n={len(correct_trials)})')
    if len(incorrect_trials) > 10:
        ax.hist(incorrect_trials['response_time'], bins=25, alpha=0.7, color='red', 
               density=True, label=f'Incorrect (n={len(incorrect_trials)})')
    
    ax.set_xlabel('Response Time (s)')
    ax.set_ylabel('Density')
    ax.set_title('Response Time Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Accuracy by Condition
    ax = axes[0, 1]
    if len(df['condition'].unique()) > 1:
        condition_acc = df.groupby('condition')['accuracy'].agg(['mean', 'sem', 'count']).reset_index()
        
        # Plot observed data
        ax.errorbar(condition_acc['condition'], condition_acc['mean'], 
                   yerr=condition_acc['sem'], marker='o', linewidth=2, 
                   markersize=8, label='Observed', color='blue')
        
        # Plot model predictions if available
        if predictions:
            pred_conditions = []
            pred_accuracies = []
            for cond, pred in predictions.items():
                if pred is not None and pred.get('pred_correct_prob') is not None:
                    pred_conditions.append(cond)
                    pred_accuracies.append(pred['pred_correct_prob'])
            
            if pred_conditions:
                ax.plot(pred_conditions, pred_accuracies, 's--', 
                       linewidth=2, markersize=6, label='Model', color='red')
        
        ax.set_xlabel('Condition')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by Condition')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.bar(['Overall'], [df['accuracy'].mean()], color='skyblue', alpha=0.7)
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Overall Accuracy: {df["accuracy"].mean():.3f}')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    
    # 3. RT by Condition and Accuracy
    ax = axes[0, 2]
    if len(df['condition'].unique()) > 1:
        conditions = sorted(df['condition'].unique())
        correct_rts = []
        incorrect_rts = []
        
        for condition in conditions:
            cond_data = df[df['condition'] == condition]
            correct_rt = cond_data[cond_data['accuracy'] == 1]['response_time'].mean()
            incorrect_rt = cond_data[cond_data['accuracy'] == 0]['response_time'].mean()
            correct_rts.append(correct_rt)
            incorrect_rts.append(incorrect_rt)
        
        ax.plot(conditions, correct_rts, 'o-', label='Correct', color='green', linewidth=2, markersize=8)
        ax.plot(conditions, incorrect_rts, 's-', label='Incorrect', color='red', linewidth=2, markersize=8)
        
        ax.set_xlabel('Condition')
        ax.set_ylabel('Mean RT (s)')
        ax.set_title('Response Time by Condition')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        correct_rt = df[df['accuracy'] == 1]['response_time'].mean()
        incorrect_rt = df[df['accuracy'] == 0]['response_time'].mean()
        ax.bar(['Correct', 'Incorrect'], [correct_rt, incorrect_rt], 
               color=['green', 'red'], alpha=0.7)
        ax.set_ylabel('Mean RT (s)')
        ax.set_title('Response Time by Accuracy')
        ax.grid(True, alpha=0.3)
    
    # 4. Parameter Values
    ax = axes[1, 0]
    param_names = ['Drift Base', 'Drift Scale', 'Boundary', 'Non-Decision Time']
    param_keys = ['drift_base', 'drift_scale', 'B', 'nondectime']
    param_values = [fitted_params.get(key, 0) for key in param_keys]
    colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
    
    bars = ax.bar(param_names, param_values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Parameter Value')
    ax.set_title('Fitted DDM Parameters')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, param_values):
        if not np.isnan(value):
            ax.text(bar.get_x() + bar.get_width()/2, 
                   bar.get_height() + max(param_values) * 0.02, 
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Model Trajectories
    ax = axes[1, 1]
    if fitted_params:
        drift = fitted_params.get('drift_base', 0.5) + fitted_params.get('drift_scale', 1.0)
        boundary = fitted_params.get('B', 1.0)
        ndt = fitted_params.get('nondectime', 0.2)
        
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
    
    # 6. Summary Statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate summary statistics
    n_trials = len(df)
    n_conditions = len(df['condition'].unique())
    overall_accuracy = df['accuracy'].mean()
    mean_rt = df['response_time'].mean()
    mean_correct_rt = df[df['accuracy'] == 1]['response_time'].mean()
    mean_incorrect_rt = df[df['accuracy'] == 0]['response_time'].mean()
    
    summary_text = f"""ANALYSIS SUMMARY

Dataset:
• Total trials: {n_trials:,}
• Conditions: {n_conditions}
• Overall accuracy: {overall_accuracy:.3f}

Response Times:
• Mean RT: {mean_rt:.3f}s
• Correct RT: {mean_correct_rt:.3f}s
• Incorrect RT: {mean_incorrect_rt:.3f}s

Model Parameters:
• Drift base: {fitted_params.get('drift_base', 'N/A'):.3f}
• Drift scale: {fitted_params.get('drift_scale', 'N/A'):.3f}  
• Boundary: {fitted_params.get('B', 'N/A'):.3f}
• Non-decision time: {fitted_params.get('nondectime', 'N/A'):.3f}s

Model Quality:
• Fitting: {'Success' if fitted_params else 'Failed'}
• Predictions: {'Available' if predictions else 'N/A'}
"""
    
    ax.text(0.05, 0.95, summary_text.strip(), transform=ax.transAxes,
           verticalalignment='top', fontsize=10, fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    return fig

def estimate_basic_ddm_parameters(df):
    """Estimates DDM parameters using basic methods when PyDDM is not available."""
    print(f"\nBasic DDM Parameter Estimation (without PyDDM):")
    
    # Group by condition
    conditions = sorted(df['condition'].unique())
    all_params = {}
    
    for condition in conditions:
        cond_data = df[df['condition'] == condition]
        
        if len(cond_data) < 10:
            continue
            
        # Basic parameter estimation
        accuracy = cond_data['accuracy'].mean()
        mean_rt = cond_data['response_time'].mean()
        std_rt = cond_data['response_time'].std()
        
        # Estimate parameters using method of moments
        ndt_est = max(0.1, mean_rt * 0.3)  # Rough estimate
        boundary_est = max(0.5, std_rt * 2)  # Rough estimate
        
        # Drift rate from accuracy
        if accuracy > 0.5:
            drift_est = 2 * (accuracy - 0.5) * boundary_est / mean_rt
        else:
            drift_est = -2 * (0.5 - accuracy) * boundary_est / mean_rt
            
        all_params[condition] = {
            'drift': drift_est,
            'boundary': boundary_est, 
            'ndt': ndt_est,
            'accuracy': accuracy,
            'mean_rt': mean_rt
        }
        
        print(f"  Condition {condition}: drift={drift_est:.3f}, boundary={boundary_est:.3f}, ndt={ndt_est:.3f}")
    
    return all_params

def create_basic_visualization(df, params_by_condition):
    """Creates visualization using basic DDM analysis."""
    print(f"\nCreating Basic DDM Visualization:")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Basic DDM Analysis Results (without PyDDM)', fontsize=16, fontweight='bold')
    
    # 1. Response Time Distribution
    ax = axes[0, 0]
    correct_trials = df[df['accuracy'] == 1]
    incorrect_trials = df[df['accuracy'] == 0]
    
    ax.hist(df['response_time'], bins=30, alpha=0.5, color='lightblue', 
           density=True, label=f'All trials (n={len(df)})')
    if len(correct_trials) > 10:
        ax.hist(correct_trials['response_time'], bins=25, alpha=0.7, color='green', 
               density=True, label=f'Correct (n={len(correct_trials)})')
    if len(incorrect_trials) > 10:
        ax.hist(incorrect_trials['response_time'], bins=25, alpha=0.7, color='red', 
               density=True, label=f'Incorrect (n={len(incorrect_trials)})')
    
    ax.set_xlabel('Response Time (s)')
    ax.set_ylabel('Density')
    ax.set_title('Response Time Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Accuracy by Condition
    ax = axes[0, 1]
    if len(df['condition'].unique()) > 1:
        condition_stats = df.groupby('condition')['accuracy'].agg(['mean', 'sem']).reset_index()
        
        ax.errorbar(condition_stats['condition'], condition_stats['mean'], 
                   yerr=condition_stats['sem'], marker='o', linewidth=2, 
                   markersize=8, label='Observed', color='blue')
        
        ax.set_xlabel('Condition')
        ax.set_ylabel('Accuracy') 
        ax.set_title('Accuracy by Condition')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.bar(['Overall'], [df['accuracy'].mean()], color='skyblue', alpha=0.7)
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Overall Accuracy: {df["accuracy"].mean():.3f}')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    
    # 3. RT by Condition and Accuracy
    ax = axes[0, 2]
    if len(df['condition'].unique()) > 1:
        conditions = sorted(df['condition'].unique())
        correct_rts = []
        incorrect_rts = []
        
        for condition in conditions:
            cond_data = df[df['condition'] == condition]
            correct_rt = cond_data[cond_data['accuracy'] == 1]['response_time'].mean()
            incorrect_rt = cond_data[cond_data['accuracy'] == 0]['response_time'].mean()
            correct_rts.append(correct_rt)
            incorrect_rts.append(incorrect_rt)
        
        ax.plot(conditions, correct_rts, 'o-', label='Correct', color='green', linewidth=2, markersize=8)
        ax.plot(conditions, incorrect_rts, 's-', label='Incorrect', color='red', linewidth=2, markersize=8)
        
        ax.set_xlabel('Condition')
        ax.set_ylabel('Mean RT (s)')
        ax.set_title('Response Time by Condition')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        correct_rt = df[df['accuracy'] == 1]['response_time'].mean()
        incorrect_rt = df[df['accuracy'] == 0]['response_time'].mean()
        ax.bar(['Correct', 'Incorrect'], [correct_rt, incorrect_rt], 
               color=['green', 'red'], alpha=0.7)
        ax.set_ylabel('Mean RT (s)')
        ax.set_title('Response Time by Accuracy')
        ax.grid(True, alpha=0.3)
    
    # 4. Parameter Values by Condition
    ax = axes[1, 0]
    if params_by_condition:
        conditions = list(params_by_condition.keys())
        drift_values = [params_by_condition[c]['drift'] for c in conditions]
        
        ax.plot(conditions, drift_values, 'o-', linewidth=2, markersize=8, color='red')
        ax.set_xlabel('Condition')
        ax.set_ylabel('Drift Rate')
        ax.set_title('Estimated Drift Rate by Condition')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No parameter\nestimates available', 
               transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title('DDM Parameters')
    
    # 5. Simulated Trajectories
    ax = axes[1, 1]
    if params_by_condition:
        # Use parameters from middle condition or first available
        mid_condition = list(params_by_condition.keys())[len(params_by_condition)//2]
        params = params_by_condition[mid_condition]
        
        trajectories = simulate_ddm_trajectories(
            drift=params['drift'], 
            boundary=params['boundary'], 
            ndt=params['ndt'], 
            n_trajectories=30
        )
        
        ax.axhline(params['boundary'], color='green', linewidth=2, alpha=0.8, label='Upper Boundary')
        ax.axhline(-params['boundary'], color='red', linewidth=2, alpha=0.8, label='Lower Boundary')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Starting Point')
        
        for traj in trajectories:
            color = 'green' if traj['choice'] == 1 else 'red'
            alpha = 0.6 if traj['decided'] else 0.3
            ax.plot(traj['time'], traj['evidence'], color=color, alpha=alpha, linewidth=1)
        
        ax.set_xlim(0, 2)
        ax.set_ylim(-params['boundary']*1.2, params['boundary']*1.2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Evidence')
        ax.set_title(f'Simulated DDM Trajectories\n(Condition {mid_condition})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No parameters\navailable for simulation', 
               transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title('DDM Trajectories')
    
    # 6. Summary Statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    n_trials = len(df)
    n_conditions = len(df['condition'].unique())
    overall_accuracy = df['accuracy'].mean()
    mean_rt = df['response_time'].mean()
    mean_correct_rt = df[df['accuracy'] == 1]['response_time'].mean()
    mean_incorrect_rt = df[df['accuracy'] == 0]['response_time'].mean()
    
    summary_text = f"""ANALYSIS SUMMARY

Dataset:
• Total trials: {n_trials:,}
• Conditions: {n_conditions}
• Overall accuracy: {overall_accuracy:.3f}

Response Times:
• Mean RT: {mean_rt:.3f}s
• Correct RT: {mean_correct_rt:.3f}s
• Incorrect RT: {mean_incorrect_rt:.3f}s

Analysis Method:
• PyDDM: {'Available' if PYDDM_AVAILABLE else 'Not Available'}
• Method: {'Full PyDDM' if PYDDM_AVAILABLE else 'Basic Estimation'}

Parameter Estimates:
{f'• {len(params_by_condition)} conditions analyzed' if params_by_condition else '• No estimates available'}

Note: For full DDM analysis,
install PyDDM with:
pip install pyddm
"""
    
    ax.text(0.05, 0.95, summary_text.strip(), transform=ax.transAxes,
           verticalalignment='top', fontsize=10, fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    return fig

def run_pyddm_analysis(file_path):
    """Runs complete DDM analysis pipeline (with or without PyDDM)."""
    print(f"="*60)
    print(f"DDM CSV Analysis Tool")
    print(f"="*60)
    
    # Load and validate data
    df = load_and_validate_csv_data(file_path)
    if df is None:
        return None, None, None, None
    
    if PYDDM_AVAILABLE:
        # Full PyDDM analysis
        print(f"\nUsing PyDDM for full analysis")
        
        # Convert to PyDDM format
        print(f"\nPreparing PyDDM Sample:")
        sample = prepare_pyddm_sample(df)
        
        if sample is None:
            print(f"  Failed to create PyDDM sample, falling back to basic analysis")
            # Fall back to basic analysis
            params_by_condition = estimate_basic_ddm_parameters(df)
            fig = create_basic_visualization(df, params_by_condition)
            return df, None, params_by_condition, None
        
        # Fit model
        model, fitted_params = fit_pyddm_model(sample)
        
        # Generate predictions
        predictions = None
        if model is not None:
            predictions = generate_model_predictions(model, df)
        
        # Create visualizations
        fig = create_comprehensive_visualization(df, model, fitted_params, predictions)
        
        print(f"\n" + "="*60)
        print(f"PyDDM Analysis Complete")
        print(f"="*60)
        
        return df, model, fitted_params, predictions
    
    else:
        # Basic DDM analysis without PyDDM
        print(f"\nUsing basic DDM analysis (PyDDM not available)")
        
        # Estimate parameters using basic methods
        params_by_condition = estimate_basic_ddm_parameters(df)
        
        # Create visualizations
        fig = create_basic_visualization(df, params_by_condition)
        
        print(f"\n" + "="*60)
        print(f"Basic DDM Analysis Complete")
        print(f"="*60)
        
        return df, None, params_by_condition, None

# Example usage
if __name__ == "__main__":
    # Example file path - update this to your CSV file location
    file_path = r"C:\Users\Rishv\OneDrive\Documents\Personal\Projects\PyDDM\dual_choice_ddm_extended.csv"
    
    print("Expected CSV format:")
    print("- Required columns: 'response_time', 'accuracy'")
    print("- Optional column: 'condition' (for condition-dependent analysis)")
    print("- response_time: positive values in seconds")
    print("- accuracy: 0 (incorrect) or 1 (correct)")
    print("- condition: numeric values (e.g., coherence levels)")
    print()
    
    # Run analysis
    results = run_pyddm_analysis(file_path)
    
    if results[0] is not None:
        print("\nAnalysis completed successfully!")
        print("Close the figure window when done viewing.")
        plt.show()
    else:
        print("\nAnalysis failed. Please check your data file and try again.")
        
        # Create example dataset for demonstration
        print("\nCreating example dataset...")
        np.random.seed(42)
        n_trials = 1000
        
        example_data = []
        for condition in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for _ in range(n_trials // 5):
                # Simulate simple DDM data
                drift = 0.5 + 2.0 * condition
                rt = np.random.gamma(2, 0.3) + 0.2
                accuracy = 1 if np.random.rand() < (1 / (1 + np.exp(-drift))) else 0
                
                example_data.append({
                    'response_time': rt,
                    'accuracy': accuracy,
                    'condition': condition
                })
        
        example_df = pd.DataFrame(example_data)
        example_df.to_csv('example_ddm_data.csv', index=False)
        print("Example dataset saved as 'example_ddm_data.csv'")
        
        # Analyze example data
        print("\nRunning analysis on example data...")
        results = run_pyddm_analysis('example_ddm_data.csv')
        
        if results[0] is not None:
            print("\nExample analysis completed!")
            plt.show()