import pyddm
import matplotlib.pyplot as plt

# Define the model with specific parameters
model = pyddm.gddm(drift=0.5, noise=1.0, bound=0.6, starting_position=0.3, nondecision=0.2)

# Solve the model
solution = model.solve()

# Plot the PDF of correct responses
plt.plot(solution.t_domain, solution.pdf("correct"))
plt.xlabel("Time (s)")
plt.ylabel("Probability Density")
plt.title("Reaction Time Distribution with Custom Parameters")
plt.show()