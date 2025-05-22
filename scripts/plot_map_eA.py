import matplotlib.pyplot as plt
from colors import BLUE_EDGE, RED_EDGE


def plot_dist_gain_graphs(metrics):
    # Data
    lambda_dist = [0.01, 0.05, 0.10, 0.5, 1.0]

    # Plot setup

    for dataset, data in metrics.items():
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axs[0].set_title("mAP$_{.5}$ as a function of $\lambda_{dist}$ on " + dataset + " dataset")
        axs[0].set_ylabel("mAP$_{.5}$")
        axs[0].grid(True)
        # Plot mAP50
        for model, values in data.items():
            map50 = values["map50"]
            e_A = values["e_A"]
            axs[0].plot(lambda_dist, map50, marker="o", color=BLUE_EDGE, label=model)
            axs[1].plot(lambda_dist, e_A, marker="o", color=RED_EDGE, label=model)

        # Plot ε_A
        axs[1].set_xlabel("$\lambda_{dist}$")
        axs[1].set_ylabel("$\epsilon_A$")
        axs[1].grid(True)
        axs[1].set_title("$\epsilon_A$ as a function of $\lambda_{dist}$ on " + dataset + " dataset")

        axs[1].legend(loc="upper right")
        axs[0].legend(loc="upper right")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        plt.close()


if __name__ == "__main__":
    metrics = {
        "KITTI": {
            "yolo11n-dist": {
                "map50": [83.9, 84.6, 84.9, 83.4, 79.5],
                "e_A": [2.20, 1.60, 1.50, 1.19, 1.21],
            },
            # "yolo11n-SPDConv3-dist": {
            #    "map50": [85.3, 84.7, 84.1, 83.0, 81.9],
            #    "e_A": [2.34, 1.67, 1.53, 1.18, 1.09],
            # },
        },
        "Waymo-night": {
            "yolo11n-dist": {
                "map50": [55.2, 55.1, 54.9, 50.4, 51.0],
                "e_A": [20.6, 22.0, 22.3, 21.1, 20.6],
            },
            # "yolo11n-SPDConv3-dist": {
            #    "map50": [55.4, 55.0, 55.8, 53.8, 50.7],
            #    "e_A": [21.7, 20.8, 21.1, 21.7, 20.7],
            # },
        },
    }

    plot_dist_gain_graphs(metrics)
