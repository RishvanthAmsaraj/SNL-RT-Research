import numpy as np
import csv

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for Single-Choice DDM (Detection Task)
num_trials = 2000  # Number of trials
drift_rate = 0.70  # Further reduced drift rate for more timeouts
decision_boundary = 1.3  # Increased boundary for harder detection
starting_point = 0.0  # Always start at 0 for single-choice
non_decision_time = 0.25  # Non-decision time (encoding + motor response)
dt = 0.001  # Time step for simulation
max_time = 3.0  # Maximum time before timeout/no-response
noise_sd = 1.0  # Standard deviation of diffusion noise

# Context: Simulating a detection task, e.g., "Is there a signal present?"
# Boundary hit = detection/response (accuracy=1)
# Timeout = no detection/miss (accuracy=0)

print("🚀 GENERATING SINGLE-CHOICE DDM DATA")
print("=" * 40)
print(f"Parameters:")
print(f"  Drift rate: {drift_rate}")
print(f"  Decision boundary: {decision_boundary}")
print(f"  Non-decision time: {non_decision_time}s")
print(f"  Max time: {max_time}s")
print(f"  Noise SD: {noise_sd}")
print()

# Lists to store results
trials = []
response_times = []
accuracies = []
decision_times = []  # Time to cross boundary (excluding non-decision time)

# Simulation
detected_count = 0
timeout_count = 0

for trial in range(num_trials):
    evidence = 0.0  # Start at zero evidence
    t = 0.0
    
    # Evidence accumulation process
    while evidence < decision_boundary and t < max_time:
        # Add drift and noise
        evidence += drift_rate * dt + np.random.normal(0, noise_sd * np.sqrt(dt))
        t += dt
        
        # Reflect at 0 to prevent negative evidence (more realistic for detection tasks)
        evidence = max(evidence, 0)
    
    # Determine response
    if evidence >= decision_boundary:
        # Detection made
        decision_time = t
        response_time = t + non_decision_time
        accuracy = 1
        detected_count += 1
    else:
        # Timeout - no detection
        decision_time = max_time
        response_time = max_time + non_decision_time
        accuracy = 0
        timeout_count += 1
    
    # Store results
    trials.append(trial + 1)
    response_times.append(response_time)
    accuracies.append(accuracy)
    decision_times.append(decision_time)

# Print summary statistics
detection_rate = detected_count / num_trials
mean_rt_detected = np.mean([rt for rt, acc in zip(response_times, accuracies) if acc == 1]) if detected_count > 0 else 0
mean_rt_timeout = np.mean([rt for rt, acc in zip(response_times, accuracies) if acc == 0]) if timeout_count > 0 else 0

print(f"Simulation Results:")
print(f"  Detections: {detected_count} ({detection_rate:.1%})")
print(f"  Timeouts: {timeout_count} ({timeout_count/num_trials:.1%})")
print(f"  Mean RT (detected): {mean_rt_detected:.3f}s")
print(f"  Mean RT (timeout): {mean_rt_timeout:.3f}s")
print()

# Save extended data with decision times
extended_filename = 'single_choice_ddm_extended.csv'
with open(extended_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['trial', 'response_time', 'accuracy', 'decision_time', 'evidence_time'])
    for i in range(num_trials):
        evidence_time = decision_times[i]  # Time spent accumulating evidence
        writer.writerow([trials[i], response_times[i], accuracies[i], decision_times[i], evidence_time])

print(f"✅ Extended CSV file '{extended_filename}' generated.")
print()
print("📋 File ready for single-choice DDM analysis!")