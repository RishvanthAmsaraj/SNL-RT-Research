import numpy as np
import csv

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for Dual-Choice DDM (Two-Alternative Forced Choice Task)
num_trials = 2000  # Number of trials
drift_rate = 0.50  # Drift rate favoring upper boundary (positive choice)
upper_boundary = 1.0  # Upper decision boundary (choice A)
lower_boundary = -1.0  # Lower decision boundary (choice B)
starting_point = 0.0  # Starting point (can be biased if needed)
non_decision_time = 0.25  # Non-decision time (encoding + motor response)
dt = 0.001  # Time step for simulation
max_time = 3.0  # Maximum time before timeout/no-response
noise_sd = 1.0  # Standard deviation of diffusion noise

# Context: Simulating a two-alternative forced choice task, e.g., "Is the stimulus A or B?"
# Upper boundary hit = Choice A (accuracy depends on true answer)
# Lower boundary hit = Choice B (accuracy depends on true answer)
# Timeout = no decision made (accuracy=0)

print("🚀 GENERATING DUAL-CHOICE DDM DATA")
print("=" * 40)
print(f"Parameters:")
print(f"  Drift rate: {drift_rate}")
print(f"  Upper boundary: {upper_boundary}")
print(f"  Lower boundary: {lower_boundary}")
print(f"  Starting point: {starting_point}")
print(f"  Non-decision time: {non_decision_time}s")
print(f"  Max time: {max_time}s")
print(f"  Noise SD: {noise_sd}")
print()

# Lists to store results
trials = []
response_times = []
accuracies = []
decision_times = []  # Time to cross boundary (excluding non-decision time)
choices = []  # Which boundary was hit (1=upper, -1=lower, 0=timeout)
evidence_trajectories = []  # Optional: store final evidence values

# Simulation
upper_count = 0
lower_count = 0
timeout_count = 0

for trial in range(num_trials):
    evidence = starting_point  # Start at starting point
    t = 0.0
    
    # For this simulation, assume the "correct" answer is upper boundary 70% of the time
    # This creates a ground truth for calculating accuracy
    true_choice = 1 if np.random.random() < 0.7 else -1
    
    # Evidence accumulation process
    while evidence < upper_boundary and evidence > lower_boundary and t < max_time:
        # Add drift (toward true choice) and noise
        true_drift = drift_rate if true_choice == 1 else -drift_rate
        evidence += true_drift * dt + np.random.normal(0, noise_sd * np.sqrt(dt))
        t += dt
    
    # Determine response and accuracy
    if evidence >= upper_boundary:
        # Upper boundary hit (Choice A)
        choice = 1
        decision_time = t
        response_time = t + non_decision_time
        accuracy = 1 if true_choice == 1 else 0  # Correct if true answer was upper
        upper_count += 1
    elif evidence <= lower_boundary:
        # Lower boundary hit (Choice B)
        choice = -1
        decision_time = t
        response_time = t + non_decision_time
        accuracy = 1 if true_choice == -1 else 0  # Correct if true answer was lower
        lower_count += 1
    else:
        # Timeout - no decision
        choice = 0
        decision_time = max_time
        response_time = max_time + non_decision_time
        accuracy = 0
        timeout_count += 1
    
    # Store results
    trials.append(trial + 1)
    response_times.append(response_time)
    accuracies.append(accuracy)
    decision_times.append(decision_time)
    choices.append(choice)
    evidence_trajectories.append(evidence)

# Print summary statistics
upper_rate = upper_count / num_trials
lower_rate = lower_count / num_trials
timeout_rate = timeout_count / num_trials
overall_accuracy = np.mean(accuracies)

mean_rt_upper = np.mean([rt for rt, choice in zip(response_times, choices) if choice == 1]) if upper_count > 0 else 0
mean_rt_lower = np.mean([rt for rt, choice in zip(response_times, choices) if choice == -1]) if lower_count > 0 else 0
mean_rt_timeout = np.mean([rt for rt, choice in zip(response_times, choices) if choice == 0]) if timeout_count > 0 else 0

accuracy_upper = np.mean([acc for acc, choice in zip(accuracies, choices) if choice == 1]) if upper_count > 0 else 0
accuracy_lower = np.mean([acc for acc, choice in zip(accuracies, choices) if choice == -1]) if lower_count > 0 else 0

print(f"Simulation Results:")
print(f"  Upper boundary hits: {upper_count} ({upper_rate:.1%})")
print(f"  Lower boundary hits: {lower_count} ({lower_rate:.1%})")
print(f"  Timeouts: {timeout_count} ({timeout_rate:.1%})")
print(f"  Overall accuracy: {overall_accuracy:.1%}")
print()
print(f"Response Times:")
print(f"  Mean RT (upper): {mean_rt_upper:.3f}s")
print(f"  Mean RT (lower): {mean_rt_lower:.3f}s")
print(f"  Mean RT (timeout): {mean_rt_timeout:.3f}s")
print()
print(f"Accuracy by Choice:")
print(f"  Accuracy (upper): {accuracy_upper:.1%}")
print(f"  Accuracy (lower): {accuracy_lower:.1%}")
print()

# Save extended data with decision times and choices
extended_filename = 'dual_choice_ddm_extended.csv'
with open(extended_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['trial', 'response_time', 'accuracy', 'decision_time', 'choice', 'final_evidence'])
    for i in range(num_trials):
        writer.writerow([
            trials[i], 
            response_times[i], 
            accuracies[i], 
            decision_times[i], 
            choices[i],
            evidence_trajectories[i]
        ])

print(f"✅ Extended CSV file '{extended_filename}' generated.")
print()
print("📋 File ready for dual-choice DDM analysis!")
print()
print("Choice Encoding:")
print("  1 = Upper boundary hit (Choice A)")
print("  -1 = Lower boundary hit (Choice B)")
print("  0 = Timeout (no decision)")