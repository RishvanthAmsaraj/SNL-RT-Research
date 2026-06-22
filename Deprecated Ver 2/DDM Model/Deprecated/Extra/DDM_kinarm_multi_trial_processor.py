"""
Kinarm Multi-Trial DDM Processor
This script processes multiple Kinarm trials from STAT and INTER blocks
and fits Drift Diffusion Models to the reaction time data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from ddm import Model, Sample, Fittable
    from ddm.functions import fit_adjust_model, display_model
    from ddm.models import DriftConstant, NoiseConstant, BoundConstant, OverlayNonDecision, ICPointSourceCenter
    PYDDM_AVAILABLE = True
except ImportError:
    print("Warning: PyDDM not installed. Install with: pip install pyddm --break-system-packages")
    PYDDM_AVAILABLE = False


class KinarmDataProcessor:
    """Process Kinarm trial data from multiple blocks."""
    
    def __init__(self, base_path):
        """
        Initialize processor with base path to data.
        
        Parameters:
        -----------
        base_path : str or Path
            Path to directory containing block folders (e.g., CMT003_01_STAT.kinarm)
        """
        self.base_path = Path(base_path)
        self.trial_data = []
        self.rt_summary = None
        
    def extract_reaction_time(self, trial_df):
        """
        Extract reaction time from a single trial CSV.
        
        Reaction time is calculated as the time from target appearance 
        to first movement onset.
        
        Parameters:
        -----------
        trial_df : pd.DataFrame
            Single trial data
            
        Returns:
        --------
        float or None : Reaction time in seconds, or None if cannot be determined
        """
        # Look for movement onset - when velocity exceeds threshold
        if 'Right_HandVelX' in trial_df.columns and 'Right_HandVelY' in trial_df.columns:
            # Calculate magnitude of velocity
            vel_mag = np.sqrt(trial_df['Right_HandVelX']**2 + trial_df['Right_HandVelY']**2)
            
            # Find first frame where velocity exceeds threshold (e.g., 0.05 m/s)
            vel_threshold = 0.05
            movement_onset = vel_mag[vel_mag > vel_threshold].index
            
            if len(movement_onset) > 0:
                onset_frame = movement_onset[0]
                # Assuming 100 Hz sampling (10ms per frame)
                rt_seconds = onset_frame * 0.01
                return rt_seconds
        
        # Alternative: look for TGT_REACHED column if available
        if 'TGT_REACHED' in trial_df.columns:
            reached_frames = trial_df[trial_df['TGT_REACHED'].notna()]
            if len(reached_frames) > 0:
                first_reach = reached_frames.index[0]
                return first_reach * 0.01
        
        return None
    
    def parse_trial_filename(self, filename):
        """
        Parse trial filename to extract trial number and condition.
        
        Example: Trial3.TP14.C2.csv -> trial=3, tp=14, condition=2
        
        Parameters:
        -----------
        filename : str
            Trial filename
            
        Returns:
        --------
        dict : Parsed information
        """
        parts = filename.replace('.csv', '').split('.')
        info = {}
        
        for part in parts:
            if part.startswith('Trial'):
                info['trial_num'] = int(part.replace('Trial', ''))
            elif part.startswith('TP'):
                info['tp'] = int(part.replace('TP', ''))
            elif part.startswith('C'):
                info['condition'] = int(part.replace('C', ''))
        
        return info
    
    def parse_block_folder(self, folder_name):
        """
        Parse block folder name to extract subject, block number, and type.
        
        Example: CMT003_01_STAT.kinarm -> subject=CMT003, block=1, type=STAT
        
        Parameters:
        -----------
        folder_name : str
            Block folder name
            
        Returns:
        --------
        dict : Parsed information
        """
        parts = folder_name.replace('.kinarm', '').split('_')
        
        info = {
            'subject': parts[0] if len(parts) > 0 else None,
            'block_num': int(parts[1]) if len(parts) > 1 else None,
            'block_type': parts[2] if len(parts) > 2 else None
        }
        
        return info
    
    def load_trials_from_block(self, block_folder):
        """
        Load all trials from a single block folder.
        
        Parameters:
        -----------
        block_folder : Path
            Path to block folder
            
        Returns:
        --------
        list : List of trial dictionaries with metadata and RT
        """
        block_info = self.parse_block_folder(block_folder.name)
        trials = []
        
        # Find all CSV files in block folder
        csv_files = sorted(block_folder.glob('*.csv'))
        
        print(f"Processing block {block_folder.name}: found {len(csv_files)} trials")
        
        for csv_file in csv_files:
            try:
                # Load trial data
                trial_df = pd.read_csv(csv_file)
                
                # Extract reaction time
                rt = self.extract_reaction_time(trial_df)
                
                if rt is not None:
                    # Parse trial filename
                    trial_info = self.parse_trial_filename(csv_file.name)
                    
                    # Combine all info
                    trial_record = {
                        **block_info,
                        **trial_info,
                        'rt_seconds': rt,
                        'rt_ms': rt * 1000,
                        'filename': csv_file.name
                    }
                    
                    trials.append(trial_record)
            
            except Exception as e:
                print(f"  Warning: Could not process {csv_file.name}: {e}")
        
        return trials
    
    def load_all_trials(self, subject_pattern='CMT003'):
        """
        Load all trials from all blocks matching subject pattern.
        
        Parameters:
        -----------
        subject_pattern : str
            Pattern to match subject folders (e.g., 'CMT003')
        """
        # Find all block folders
        block_folders = sorted(self.base_path.glob(f'{subject_pattern}*'))
        
        print(f"Found {len(block_folders)} block folders")
        
        all_trials = []
        for block_folder in block_folders:
            if block_folder.is_dir():
                trials = self.load_trials_from_block(block_folder)
                all_trials.extend(trials)
        
        self.trial_data = pd.DataFrame(all_trials)
        
        # Add hand and speed labels
        self.add_condition_labels()
        
        print(f"\nTotal trials loaded: {len(self.trial_data)}")
        print(f"Blocks: {self.trial_data['block_type'].value_counts().to_dict()}")
        
        return self.trial_data
    
    def add_condition_labels(self):
        """Add hand and speed labels based on TP codes."""
        if self.trial_data is None or len(self.trial_data) == 0:
            return
        
        # STAT blocks: TP13=Right, TP14=Left
        # INTER blocks: TP5=Right 75deg/s, TP6=Left 75deg/s, TP9=Right 150deg/s, TP10=Left 150deg/s
        
        def get_hand_speed(row):
            tp = row['tp']
            block_type = row['block_type']
            
            if block_type == 'STAT':
                if tp == 13:
                    return 'Right', 0
                elif tp == 14:
                    return 'Left', 0
            elif block_type == 'INTER':
                if tp == 5:
                    return 'Right', 75
                elif tp == 6:
                    return 'Left', 75
                elif tp == 9:
                    return 'Right', 150
                elif tp == 10:
                    return 'Left', 150
            
            return None, None
        
        self.trial_data[['hand', 'speed']] = self.trial_data.apply(
            lambda row: pd.Series(get_hand_speed(row)), axis=1
        )
        
        # Create condition label
        self.trial_data['condition'] = self.trial_data.apply(
            lambda row: f"{row['block_type']}_{row['hand']}_{row['speed']}deg" 
            if pd.notna(row['hand']) else None,
            axis=1
        )
    
    def create_summary(self):
        """Create summary statistics by condition."""
        if self.trial_data is None or len(self.trial_data) == 0:
            print("No data loaded")
            return None
        
        summary = self.trial_data.groupby(['block_type', 'hand', 'speed']).agg({
            'rt_ms': ['count', 'mean', 'std', 'median', 'min', 'max']
        }).round(2)
        
        self.rt_summary = summary
        return summary
    
    def plot_rt_distributions(self, save_path=None):
        """
        Plot reaction time distributions by condition.
        
        Parameters:
        -----------
        save_path : str or None
            If provided, save figure to this path
        """
        if self.trial_data is None or len(self.trial_data) == 0:
            print("No data to plot")
            return
        
        # STAT blocks
        stat_data = self.trial_data[self.trial_data['block_type'] == 'STAT']
        
        if len(stat_data) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # STAT - Right vs Left
            stat_right = stat_data[stat_data['hand'] == 'Right']['rt_ms']
            stat_left = stat_data[stat_data['hand'] == 'Left']['rt_ms']
            
            if len(stat_right) > 0 and len(stat_left) > 0:
                bins = np.linspace(
                    min(stat_right.min(), stat_left.min()),
                    max(stat_right.max(), stat_left.max()),
                    30
                )
                
                axes[0].hist(stat_right, bins=bins, alpha=0.6, label='Right', color='blue', edgecolor='black')
                axes[0].hist(stat_left, bins=bins, alpha=0.6, label='Left', color='red', edgecolor='black')
                axes[0].set_xlabel('Reaction Time (ms)')
                axes[0].set_ylabel('Count')
                axes[0].set_title('STAT Blocks: RT Distribution by Hand')
                axes[0].legend()
                axes[0].grid(alpha=0.3)
            
            # Combined STAT
            axes[1].hist(stat_data['rt_ms'], bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[1].set_xlabel('Reaction Time (ms)')
            axes[1].set_ylabel('Count')
            axes[1].set_title('STAT Blocks: Combined RT Distribution')
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}_stat.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        # INTER blocks
        inter_data = self.trial_data[self.trial_data['block_type'] == 'INTER']
        
        if len(inter_data) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            conditions = [
                ('Right', 75, 'Right 75°/s'),
                ('Left', 75, 'Left 75°/s'),
                ('Right', 150, 'Right 150°/s'),
                ('Left', 150, 'Left 150°/s')
            ]
            
            for idx, (hand, speed, label) in enumerate(conditions):
                ax = axes[idx // 2, idx % 2]
                data = inter_data[(inter_data['hand'] == hand) & (inter_data['speed'] == speed)]['rt_ms']
                
                if len(data) > 0:
                    ax.hist(data, bins=20, alpha=0.7, edgecolor='black')
                    ax.set_xlabel('Reaction Time (ms)')
                    ax.set_ylabel('Count')
                    ax.set_title(f'INTER: {label}')
                    ax.grid(alpha=0.3)
                    ax.text(0.95, 0.95, f'n={len(data)}\nμ={data.mean():.1f}ms',
                           transform=ax.transAxes, ha='right', va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}_inter.png", dpi=300, bbox_inches='tight')
            plt.show()


class DDMAnalyzer:
    """Fit Drift Diffusion Models to reaction time data."""
    
    def __init__(self, trial_data):
        """
        Initialize DDM analyzer.
        
        Parameters:
        -----------
        trial_data : pd.DataFrame
            Trial data with RT information
        """
        self.trial_data = trial_data
        self.models = {}
        self.results = []
    
    def prepare_sample(self, condition_filter):
        """
        Prepare PyDDM Sample object from filtered data.
        
        Parameters:
        -----------
        condition_filter : dict
            Dictionary of column:value pairs to filter data
            
        Returns:
        --------
        Sample : PyDDM Sample object
        """
        # Filter data
        filtered = self.trial_data.copy()
        for col, val in condition_filter.items():
            filtered = filtered[filtered[col] == val]
        
        # Get RTs in seconds
        rts = filtered['rt_seconds'].values
        
        # For DDM, we need correct/error coding
        # Since we don't have error trials in the data structure,
        # we'll assume all are correct (choice = 1)
        choices = np.ones(len(rts))
        
        # Create Sample
        sample = Sample.from_numpy_array(rts, choices)
        
        return sample, len(rts)
    
    def fit_simple_ddm(self, sample, label):
        """
        Fit a simple 3-parameter DDM.
        
        Parameters:
        -----------
        sample : Sample
            PyDDM sample object
        label : str
            Label for this condition
            
        Returns:
        --------
        Model : Fitted PyDDM model
        """
        # Define model with 3 parameters: drift, bound, non-decision time
        model = Model(
            name=label,
            drift=DriftConstant(drift=Fittable(minval=-10, maxval=20)),
            noise=NoiseConstant(noise=1),  # Fixed
            bound=BoundConstant(B=Fittable(minval=0.2, maxval=3)),
            IC=ICPointSourceCenter(),
            overlay=OverlayNonDecision(nondectime=Fittable(minval=0.1, maxval=1))
        )
        
        # Fit model
        print(f"Fitting DDM for {label}...")
        fitted_model = fit_adjust_model(sample=sample, model=model, verbose=False)
        
        return fitted_model
    
    def extract_parameters(self, model):
        """
        Extract fitted parameters from model.
        
        Parameters:
        -----------
        model : Model
            Fitted PyDDM model
            
        Returns:
        --------
        dict : Parameter values
        """
        params = {}
        
        # Get drift
        if hasattr(model.get_dependence('drift'), 'drift'):
            drift_param = model.get_dependence('drift').drift
            params['drift'] = drift_param.real() if hasattr(drift_param, 'real') else float(drift_param)
        
        # Get bound
        if hasattr(model.get_dependence('bound'), 'B'):
            bound_param = model.get_dependence('bound').B
            params['bound'] = bound_param.real() if hasattr(bound_param, 'real') else float(bound_param)
        
        # Get non-decision time
        if hasattr(model.get_dependence('overlay'), 'nondectime'):
            ndt_param = model.get_dependence('overlay').nondectime
            params['nondectime'] = ndt_param.real() if hasattr(ndt_param, 'real') else float(ndt_param)
        
        return params
    
    def fit_all_conditions(self):
        """Fit DDM to all conditions in the dataset."""
        if not PYDDM_AVAILABLE:
            print("PyDDM not available. Cannot fit models.")
            return None
        
        # Define conditions to fit
        conditions = []
        
        # STAT blocks - pooled left and right
        stat_data = self.trial_data[self.trial_data['block_type'] == 'STAT']
        if len(stat_data) > 0:
            conditions.append({
                'filter': {'block_type': 'STAT'},
                'label': 'STAT_pooled',
                'description': 'Stationary targets (both hands)'
            })
        
        # INTER blocks by speed
        inter_speeds = self.trial_data[self.trial_data['block_type'] == 'INTER']['speed'].unique()
        for speed in sorted(inter_speeds):
            if pd.notna(speed):
                conditions.append({
                    'filter': {'block_type': 'INTER', 'speed': speed},
                    'label': f'INTER_{int(speed)}deg',
                    'description': f'Interception {int(speed)}°/s (both hands)'
                })
        
        # Fit each condition
        for cond in conditions:
            try:
                sample, n_trials = self.prepare_sample(cond['filter'])
                
                if n_trials < 10:
                    print(f"Skipping {cond['label']}: insufficient trials (n={n_trials})")
                    continue
                
                print(f"\n{cond['description']} (n={n_trials})")
                model = self.fit_simple_ddm(sample, cond['label'])
                
                # Store model
                self.models[cond['label']] = model
                
                # Extract parameters
                params = self.extract_parameters(model)
                result = {
                    'Condition': cond['description'],
                    'n_trials': n_trials,
                    **params
                }
                self.results.append(result)
                
                print(f"  drift={params.get('drift', np.nan):.3f}, "
                      f"bound={params.get('bound', np.nan):.3f}, "
                      f"nondectime={params.get('nondectime', np.nan):.3f}")
                
            except Exception as e:
                print(f"Error fitting {cond['label']}: {e}")
        
        # Create results dataframe
        if len(self.results) > 0:
            results_df = pd.DataFrame(self.results)
            return results_df
        
        return None


def main():
    """Main execution function."""
    print("="*60)
    print("Kinarm Multi-Trial DDM Processor")
    print("="*60)
    
    # Set your data path here
    # This should point to the directory containing block folders
    # Example: /Users/yourname/Downloads/CMT003_PRE/CMT003_PRE/
    data_path = input("\nEnter path to data directory containing block folders: ").strip()
    
    if not data_path:
        print("Using example path (update this in the script)")
        data_path = "/path/to/your/data"
    
    # Initialize processor
    processor = KinarmDataProcessor(data_path)
    
    # Load all trials
    print("\n" + "="*60)
    print("STEP 1: Loading trial data")
    print("="*60)
    trial_data = processor.load_all_trials(subject_pattern='CMT003')
    
    # Create summary
    print("\n" + "="*60)
    print("STEP 2: Creating summary statistics")
    print("="*60)
    summary = processor.create_summary()
    print("\nSummary by condition:")
    print(summary)
    
    # Save trial data to CSV
    output_file = 'kinarm_trial_summary.csv'
    trial_data.to_csv(output_file, index=False)
    print(f"\nTrial data saved to: {output_file}")
    
    # Plot distributions
    print("\n" + "="*60)
    print("STEP 3: Plotting RT distributions")
    print("="*60)
    processor.plot_rt_distributions(save_path='kinarm_rt_dist')
    
    # Fit DDM models
    print("\n" + "="*60)
    print("STEP 4: Fitting DDM models")
    print("="*60)
    
    if PYDDM_AVAILABLE:
        analyzer = DDMAnalyzer(trial_data)
        results_df = analyzer.fit_all_conditions()
        
        if results_df is not None:
            print("\n" + "="*60)
            print("DDM RESULTS SUMMARY")
            print("="*60)
            print(results_df.to_string(index=False))
            
            # Save results
            results_df.to_csv('kinarm_ddm_results.csv', index=False)
            print(f"\nDDM results saved to: kinarm_ddm_results.csv")
    else:
        print("\nInstall PyDDM to run DDM analysis:")
        print("  pip install pyddm --break-system-packages")
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


if __name__ == '__main__':
    main()
