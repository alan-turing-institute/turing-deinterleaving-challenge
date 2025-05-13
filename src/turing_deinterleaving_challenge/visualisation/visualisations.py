import distinctipy
import matplotlib.pyplot as plt
import numpy as np

from ..data import PulseTrain

def scatter_features(
    features,  # Float[Array, "seq_len 2"],
    labels,  # Float[Array, " seq_len"],
    x_label: str,
    y_label: str,
    title: str,
    size: int | None = None,
) -> None:
    ax = plt.gca()
    emitter_labels = np.unique(labels)
    colors = distinctipy.get_colors(len(emitter_labels), pastel_factor=0.7, rng=46)
    tracks = [features[labels == label] for label in emitter_labels]

    for label, track, color in zip(emitter_labels, tracks, colors, strict=True):
        ax.scatter(
            track[:, 0],
            track[:, 1],
            s=size,
            color=color,
            label="Emitter: " + str(int(label)),
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)


def plot_true_vs_predicted_features(
    features,  # Float[Array, "seq_len 2"],
    labels_pred,  # Float[Array, " seq_len"],
    labels_true,  # Float[Array, " seq_len"],
    x_label: str,
    y_label: str,
) -> tuple[plt.Axes, plt.Axes]:
    ax1 = plt.subplot(1, 2, 1)
    scatter_features(features, labels_true, x_label, y_label, "True Clustering")

    ax2 = plt.subplot(1, 2, 2)
    scatter_features(features, labels_pred, x_label, y_label, "Predicted Clustering")

    return ax1, ax2


def plot_pdws(
    x_feature: str,
    y_feature: str,
    pulse_train,  # Float[Array, "seq_len 5"],
    labels_true,  # Float[Array, " seq_len"],
    labels_pred,  # Float[Array, " seq_len"],
) -> tuple[plt.Axes, plt.Axes]:
    features_indices = {
        "toa": 0,
        "f": 1,
        "pw": 2,
        "a": 3,
        "aoa": 4,
        "ToA": 0,
        "F": 1,
        "PW": 2,
        "A": 3,
        "AoA": 4,
        "time_of_arrival": 0,
        "frequency": 1,
        "pulse_width": 2,
        "amplitude": 3,
        "angle_of_arrival": 4,
        "Time of Arrival": 0,
        "Frequency": 1,
        "Pulse Width": 2,
        "Amplitude": 3,
        "Angle of Arrival": 4,
    }
    feature_axis_labels = {
        0: "Time of Arrival",
        1: "Frequency",
        2: "Pulse Width",
        3: "Amplitude",
        4: "Angle of Arrival",
    }

    x_feature_index = features_indices[x_feature]
    y_feature_index = features_indices[y_feature]
    features = pulse_train[:, [x_feature_index, y_feature_index]]

    x_axis_label = feature_axis_labels[x_feature_index]
    y_axis_label = feature_axis_labels[y_feature_index]

    ax1, ax2 = plot_true_vs_predicted_features(
        features, labels_pred, labels_true, x_axis_label, y_axis_label
    )
    return ax1, ax2

def plot_data(data, labels):
    unique_labels = np.unique(labels)
    label_indices = {label: i for i, label in enumerate(unique_labels)}
    num_colors = len(unique_labels)

    colors = distinctipy.get_colors(num_colors)
    color_map = [colors[label_indices[label]] for label in labels.squeeze()]
    labels = ["Frequency (MHz)", "Pulse Width (us)", "Angle of Arrival (deg)", "Amplitude (dBm)"]
    plt.figure(figsize=(16,8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.xlabel("Time of Arrival (us)")
        plt.ylabel(labels[i])
        plt.scatter(data[:, 0], data[:, i+1], c=color_map, s=10)

def plot_pulse_train(pulse_train: PulseTrain):
    """
    Plot the features of a pulse train.
    """
    plt.figure(figsize=(12, 8))
    features = pulse_train.data
    labels = pulse_train.labels.squeeze()

    plot_data(features, labels)
    plt.suptitle("Pulse Train Features")
    plt.show()
