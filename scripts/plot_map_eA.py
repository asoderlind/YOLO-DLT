import matplotlib.pyplot as plt
from colors import BLUE_EDGE, RED_EDGE


def plot_dist_gain_graphs(metrics):
    # Data
    lambda_dist_map = [0.0, 0.01, 0.05, 0.10, 0.5, 1.0]
    lambda_dist_eA = [0.01, 0.05, 0.10, 0.5, 1.0]

    # Plot setup

    for dataset, data in metrics.items():
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        if dataset == "KITTI":
            title = "mAP$_{.5}$ as a function of $\lambda_{dist}$ on " + dataset + " dataset"
            ylabel = "mAP$_{.5}$"
        else:
            title = "AP$_{.5}$ as a function of $\lambda_{dist}$ on " + dataset + " dataset"
            ylabel = "AP$_{.5}$"

        axs[0].set_title(title, fontsize=16)
        axs[0].set_ylabel(ylabel, fontsize=16)
        axs[0].grid(True)

        # Plot mAP50
        for model, values in data.items():
            map50 = values["map50"]
            e_A = values["e_A"]
            axs[0].plot(lambda_dist_map, map50, marker="o", color=BLUE_EDGE, label=model)
            axs[1].plot(lambda_dist_eA, e_A, marker="o", color=RED_EDGE, label=model)

        # Plot ε_A
        axs[1].set_xlabel("$\lambda_{dist}$", fontsize=16)
        axs[1].set_ylabel("$\epsilon_A$", fontsize=16)
        axs[1].grid(True)
        axs[1].set_title("$\epsilon_A$ as a function of $\lambda_{dist}$ on " + dataset + " dataset", fontsize=16)

        axs[1].legend(loc="center right", fontsize=12)
        axs[0].legend(loc="center right", fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        fig.set_size_inches(10, 7)
        plt.savefig(f"dist_gain_{dataset}.png", dpi=300)
        plt.show()
        plt.close()


if __name__ == "__main__":
    metrics = {
        "KITTI": {
            "YOLO11n-D": {
                "map50": [84.23, 83.90, 84.63, 84.93, 83.43, 79.53],
                "e_A": [2.20, 1.60, 1.50, 1.19, 1.21],
            },
        },
        "Waymo-night": {
            "YOLO11n-DL": {
                "map50": [64.57, 63.90, 64.20, 64.07, 62.80, 62.27],
                "e_A": [19.35, 19.47, 19.73, 19.83, 19.97],
            },
        },
    }

    plot_dist_gain_graphs(metrics)
