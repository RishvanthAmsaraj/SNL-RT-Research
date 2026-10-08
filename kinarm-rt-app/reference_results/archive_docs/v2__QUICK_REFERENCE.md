# Quick Reference: TP Codes and Conditions

## Block Naming Convention

Block folder names follow this pattern:
```
[SUBJECT]_[BLOCK_NUMBER]_[BLOCK_TYPE].kinarm
```

Examples:
- `CMT003_01_STAT.kinarm` → Subject CMT003, Block 1, Stationary type
- `CMT003_05_INTER.kinarm` → Subject CMT003, Block 5, Interception type

## TP Code Reference

### STAT Blocks (Stationary Targets)
| TP Code | Hand  | Speed    | Description                |
|---------|-------|----------|----------------------------|
| TP13    | Right | 0°/s     | Right hand, static target  |
| TP14    | Left  | 0°/s     | Left hand, static target   |

### INTER Blocks (Intercepting Targets)
| TP Code | Hand  | Speed    | Description                        |
|---------|-------|----------|-------------------------------------|
| TP5     | Right | 75°/s    | Right hand, slow moving target     |
| TP6     | Left  | 75°/s    | Left hand, slow moving target      |
| TP9     | Right | 150°/s   | Right hand, fast moving target     |
| TP10    | Left  | 150°/s   | Left hand, fast moving target      |

## Subject CMT003 Block Structure

### STAT Blocks (4 total)
- **Block 01** (CMT003_01_STAT.kinarm): ~30 trials, mix of TP13 & TP14
- **Block 02** (CMT003_02_STAT.kinarm): ~30 trials, mix of TP13 & TP14
- **Block 03** (CMT003_03_STAT.kinarm): ~30 trials, mix of TP13 & TP14
- **Block 04** (CMT003_04_STAT.kinarm): ~30 trials, mix of TP13 & TP14

**Total STAT trials**: ~120 trials (60 per hand)

### INTER Blocks (12 total)
- **Block 05** (CMT003_05_INTER.kinarm): ~30 trials, TP5/TP6/TP9/TP10
- **Block 06** (CMT003_06_INTER.kinarm): ~30 trials, TP5/TP6/TP9/TP10
- **Block 07** (CMT003_07_INTER.kinarm): ~30 trials, TP5/TP6/TP9/TP10
- **Block 08** (CMT003_08_INTER.kinarm): ~30 trials, TP5/TP6/TP9/TP10
- **Block 09** (CMT003_09_INTER.kinarm): ~30 trials, TP5/TP6/TP9/TP10
- **Block 10** (CMT003_10_INTER.kinarm): ~30 trials, TP5/TP6/TP9/TP10
- **Block 11** (CMT003_11_INTER.kinarm): ~30 trials, TP5/TP6/TP9/TP10
- **Block 12** (CMT003_12_INTER.kinarm): ~30 trials, TP5/TP6/TP9/TP10
- **Block 13** (CMT003_13_INTER.kinarm): ~30 trials, TP5/TP6/TP9/TP10
- **Block 14** (CMT003_14_INTER.kinarm): ~30 trials, TP5/TP6/TP9/TP10
- **Block 15** (CMT003_15_INTER.kinarm): ~30 trials, TP5/TP6/TP9/TP10
- **Block 16** (CMT003_16_INTER.kinarm): ~30 trials, TP5/TP6/TP9/TP10

**Total INTER trials**: ~360 trials (90 per condition)

## Trial Filename Convention

Trial files follow this pattern:
```
Trial[N].TP[CODE].C[CONDITION].csv
```

Example: `Trial3.TP14.C2.csv`
- **Trial3**: Trial number 3 within this block
- **TP14**: Left hand condition (STAT block)
- **C2**: Condition variant 2

## Data File Structure

Each trial CSV contains time-series data with columns including:
- **Frame**: Frame number (100 Hz sampling = 0.01s per frame)
- **Right_HandX, Right_HandY**: Hand position (meters)
- **Right_HandVelX, Right_HandVelY**: Hand velocity (m/s)
- **xT, yT**: Target position
- **TGT_REACHED**: Target reached marker
- **TRICEPS EMG, DELTOID EMG, PECTORALIS EMG**: Muscle activity
- **Gaze_X, Gaze_Y**: Gaze position

## Expected Output Summary

After processing all blocks, you should have:

### STAT Condition
- **Trials**: ~240 total (120 right hand + 120 left hand)
- **Expected mean RT**: 200-400ms (typical for stationary reaching)

### INTER Conditions
- **75°/s trials**: ~180 total (90 right + 90 left)
  - Expected mean RT: 250-450ms
- **150°/s trials**: ~180 total (90 right + 90 left)  
  - Expected mean RT: 200-400ms (may be faster due to urgency)

## Condition Labels in Output Files

The code creates standardized condition labels:
- `STAT_Right_0deg`: Stationary, right hand
- `STAT_Left_0deg`: Stationary, left hand
- `INTER_Right_75deg`: Interception, right hand, 75°/s
- `INTER_Left_75deg`: Interception, left hand, 75°/s
- `INTER_Right_150deg`: Interception, right hand, 150°/s
- `INTER_Left_150deg`: Interception, left hand, 150°/s

## DDM Fitting Strategy

The code fits models to pooled data:

1. **STAT_pooled**: All stationary trials (both hands combined)
2. **INTER_75deg**: All 75°/s trials (both hands combined)
3. **INTER_150deg**: All 150°/s trials (both hands combined)

This provides cleaner parameter estimates by increasing sample size per condition.

### Why Pool Across Hands?

- Increases statistical power
- Reduces noise in parameter estimates
- Allows focus on task difficulty (speed) rather than laterality
- Can be unpooled later if hand differences are of interest

## Troubleshooting TP Code Issues

If you get unexpected TP codes:
1. Check the trial filename carefully
2. Verify the block type (STAT vs INTER)
3. Look at the CSV data to confirm hand used
4. The code will print warnings for unrecognized TP codes

## Notes on Data Quality

### Good RT Indicators:
- RTs between 100-800ms for most participants
- Smooth velocity profiles
- Clear movement onset

### Potential Issues:
- RTs < 100ms: Likely anticipatory responses (may want to exclude)
- RTs > 1000ms: Possible delays or hesitations (check video if available)
- No velocity peak: Movement may not have been completed

## Customizing Analysis

To analyze specific subsets:

```python
# Only analyze one hand
trial_data = trial_data[trial_data['hand'] == 'Right']

# Only analyze one speed
trial_data = trial_data[trial_data['speed'] == 75]

# Only analyze specific blocks
trial_data = trial_data[trial_data['block_num'].isin([1, 2, 3])]

# Only analyze first 10 trials per block
trial_data = trial_data[trial_data['trial_num'] <= 10]
```

## Expected Processing Time

- **Data loading**: 30-60 seconds for 480 trials
- **RT extraction**: ~0.1 seconds per trial
- **DDM fitting**: 10-30 seconds per condition
- **Total runtime**: 2-5 minutes for complete analysis

## Verification Checklist

Before accepting results, verify:
- [ ] All expected blocks were found and loaded
- [ ] Trial counts match expectations (~30 per block)
- [ ] RT distributions look reasonable (bell-shaped, 100-800ms range)
- [ ] No major outliers or data quality issues
- [ ] DDM parameters are in reasonable ranges
  - Drift: typically 1-20
  - Bound: typically 0.2-3
  - Non-decision time: typically 0.1-0.5s
