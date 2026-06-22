# Kinarm Multi-Trial DDM Analysis

This package processes Kinarm data from multiple trials across STAT (stationary) and INTER (interception) blocks and fits Drift Diffusion Models (DDMs) to reaction time distributions.

## Overview

The Drift Diffusion Model (DDM) is a cognitive model that describes decision-making as a process of evidence accumulation. In the context of reaching tasks:

- **Drift rate (v)**: How quickly evidence accumulates toward a decision (higher = faster movement initiation)
- **Boundary (a)**: The amount of evidence needed before committing to a movement (higher = more cautious)
- **Non-decision time (t₀)**: Sensory encoding and motor execution delays

## Data Structure

Your data should be organized as follows:

```
Data Directory/
├── CMT003_01_STAT.kinarm/
│   ├── Trial1.TP13.C1.csv
│   ├── Trial2.TP14.C2.csv
│   ├── Trial3.TP13.C1.csv
│   └── ... (30 trials per block)
├── CMT003_02_STAT.kinarm/
├── CMT003_03_STAT.kinarm/
├── CMT003_04_STAT.kinarm/
├── CMT003_05_INTER.kinarm/
│   ├── Trial1.TP5.C1.csv
│   ├── Trial2.TP6.C2.csv
│   └── ... (30 trials per block)
├── CMT003_06_INTER.kinarm/
└── ... (16 blocks total)
```

### Block Types

#### STAT Blocks (4 blocks - Stationary targets)
- **TP13**: Right hand
- **TP14**: Left hand

#### INTER Blocks (12 blocks - Intercepting targets)
- **TP5**: Right hand at 75°/s
- **TP6**: Left hand at 75°/s
- **TP9**: Right hand at 150°/s
- **TP10**: Left hand at 150°/s

## Installation

### Option 1: Using the Python script

1. Install required packages:
```bash
pip install pandas numpy matplotlib pyddm --break-system-packages
```

2. Run the script:
```bash
python DDM_kinarm_multi_trial_processor.py
```

### Option 2: Using the Jupyter notebook

1. Install Jupyter and required packages:
```bash
pip install jupyter pandas numpy matplotlib pyddm --break-system-packages
```

2. Launch Jupyter:
```bash
jupyter notebook DDM_kinarm_multi_trial_analysis.ipynb
```

3. Update the `DATA_PATH` in the first code cell to point to your data directory

4. Run all cells

## What the Code Does

### 1. Data Loading
- Scans all block folders matching the subject pattern (e.g., CMT003*)
- Reads each trial CSV file
- Extracts reaction time (RT) by detecting movement onset
  - Movement onset = first frame where hand velocity exceeds 0.05 m/s

### 2. RT Extraction Method

The code calculates reaction time as:

```python
# Calculate velocity magnitude
vel_mag = sqrt(Right_HandVelX² + Right_HandVelY²)

# Find first frame where velocity > threshold (0.05 m/s)
rt_seconds = movement_onset_frame / sampling_rate
```

This gives you the time from trial start to movement initiation.

### 3. Data Organization

For each trial, the code extracts:
- Subject ID
- Block number and type (STAT/INTER)
- Trial number within block
- TP code (hand/speed condition)
- Reaction time in seconds and milliseconds
- Hand (Left/Right)
- Speed (0, 75, or 150°/s)

### 4. Analysis

The code performs:
1. **Summary statistics** by condition (mean, std, median, min, max RT)
2. **Visualization** of RT distributions
3. **DDM fitting** for each condition:
   - STAT blocks (both hands pooled)
   - INTER 75°/s (both hands pooled)
   - INTER 150°/s (both hands pooled)

## Output Files

Running the analysis generates:

1. **kinarm_trial_summary.csv** - Complete trial-level data with RTs
2. **kinarm_rt_dist_stat.png** - RT distributions for STAT blocks
3. **kinarm_rt_dist_inter.png** - RT distributions for INTER blocks (4 subplots)
4. **kinarm_ddm_results.csv** - Fitted DDM parameters by condition
5. **kinarm_ddm_parameters.png** - Bar plots of DDM parameters

## Customization

### Adjusting RT Detection

If the default velocity threshold (0.05 m/s) doesn't work well, you can adjust it:

**In the Python script:**
```python
VELOCITY_THRESHOLD = 0.05  # Change this value
```

**In the Jupyter notebook:**
```python
# In the configuration cell
VELOCITY_THRESHOLD = 0.05  # Change this value
```

### Changing Sampling Rate

If your data uses a different sampling rate (default is 100 Hz):

```python
SAMPLING_RATE = 100  # e.g., change to 200 for 200 Hz
```

### Filtering Specific Blocks

To analyze only certain blocks, modify the `load_all_trials()` function:

```python
# Only load STAT blocks
block_folders = sorted(base_path.glob(f'{subject_pattern}*STAT*'))

# Only load specific block numbers
block_folders = [f for f in block_folders if any(x in f.name for x in ['_01_', '_02_'])]
```

## Understanding the Results

### RT Distributions

The histograms show:
- **STAT blocks**: Typically faster RTs (simpler task)
- **INTER blocks**: May show slower RTs, especially at higher speeds
- **Hand differences**: Right vs Left hand performance

### DDM Parameters

Typical interpretation:
- **Higher drift rate** → Faster evidence accumulation, quicker decisions
- **Higher boundary** → More cautious, slower but potentially more accurate
- **Longer non-decision time** → Longer sensory/motor delays

Example results:
```
Condition                         n_trials  drift   bound  nondectime
Stationary (both hands)              240   12.50   0.95      0.160
Interception 75°/s (both hands)      720    9.80   0.85      0.175
Interception 150°/s (both hands)     720   15.20   1.20      0.145
```

Interpretation:
- Interception at 150°/s has higher drift rate (faster accumulation) but also higher boundary (more cautious)
- Non-decision times are relatively stable across conditions

## Troubleshooting

### Issue: "Could not extract RT"
**Solution**: Check that your CSV files contain `Right_HandVelX` and `Right_HandVelY` columns. If not, you may need to modify the RT extraction function.

### Issue: "No trials loaded"
**Solution**: 
1. Check that `DATA_PATH` is correct
2. Verify folder names match the pattern `CMT003_*`
3. Ensure CSV files follow naming convention `Trial*.TP*.C*.csv`

### Issue: PyDDM installation fails
**Solution**: Use the `--break-system-packages` flag:
```bash
pip install pyddm --break-system-packages
```

### Issue: DDM fitting fails
**Solution**: 
1. Check you have sufficient trials (>10 per condition)
2. Verify RTs are in reasonable range (not too short/long)
3. Check for outliers that might affect fitting

## Advanced Usage

### Fitting Separate Models by Hand

To fit separate models for left and right hands:

```python
# Add to conditions list
conditions.append({
    'filter': {'block_type': 'STAT', 'hand': 'Right'},
    'label': 'STAT_right',
    'description': 'Stationary Right Hand'
})
conditions.append({
    'filter': {'block_type': 'STAT', 'hand': 'Left'},
    'label': 'STAT_left',
    'description': 'Stationary Left Hand'
})
```

### Excluding Outlier RTs

To remove outliers before analysis:

```python
# Remove RTs outside 100-1000ms
trial_data = trial_data[(trial_data['rt_ms'] >= 100) & (trial_data['rt_ms'] <= 1000)]
```

### Exporting for Statistical Analysis

The `kinarm_trial_summary.csv` file can be imported into R, SPSS, or other software for additional statistical tests (ANOVA, t-tests, etc.).

## Citation

If you use this code in your research, please cite:

- **PyDDM**: Shinn, M., Lam, N.H., & Murray, J.D. (2020). A flexible framework for simulating and fitting generalized drift-diffusion models. eLife, 9, e56938.

## Contact

For questions or issues, please contact your research team or create an issue in your project repository.

## License

This code is provided for research purposes. Modify and distribute as needed for your project.
