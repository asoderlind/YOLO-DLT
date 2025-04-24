import matplotlib.pyplot as plt
from colors import BLUE_EDGE, RED_EDGE

# Data
lambda_dist = [0.01, 0.025, 0.05, 0.10, 0.175, 0.25, 0.5, 1.0, 2.0]
map50 = [88.9, 89.3, 89.7, 89.3, 88.2, 88.5, 88.3, 86.3, 82.8]
e_A = [2.44, 1.65, 1.42, 1.40, 1.35, 1.05, 0.96, 0.897, 0.904]

# Plot setup
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Plot mAP50
axs[0].plot(lambda_dist, map50, marker="o", color=BLUE_EDGE)
axs[0].set_ylabel("mAP$_{.5}$")
axs[0].grid(True)
axs[0].set_title("mAP$_{.5}$ as a function of $\lambda_{dist}$")

# Plot ε_A
axs[1].plot(lambda_dist, e_A, marker="o", color=RED_EDGE)
axs[1].set_xlabel("$\lambda_{dist}$")
axs[1].set_ylabel("$\epsilon_A$")
axs[1].grid(True)
axs[1].set_title("$\epsilon_A$ as a function of $\lambda_{dist}$")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
