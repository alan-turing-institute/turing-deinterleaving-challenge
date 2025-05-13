from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    v_measure_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import DeinterleavingChallengeDataset
from .model import Deinterleaver


def evaluate_labels(
    labels_pred,  # Int[Array, " seq_len"],
    labels_true,  # Int[Array, " seq_len"],
):
    homogeneity = homogeneity_score(labels_true, labels_pred)
    completeness = completeness_score(labels_true, labels_pred)
    v_measure = v_measure_score(labels_true, labels_pred)
    adjusted_rand_index = adjusted_rand_score(labels_true, labels_pred)
    adjusted_mutual_information = adjusted_mutual_info_score(labels_true, labels_pred)
    return {
        "Homogeneity": homogeneity,
        "Completeness": completeness,
        "V-measure": v_measure,
        "Adjusted Rand Index": adjusted_rand_index,
        "Adjusted Mutual Information": adjusted_mutual_information,
    }


def evaluate_model_on_pulse_train(
    model: Deinterleaver,
    pulse_train,  # Float[Array, "seq_len feature_len"],
    labels_true,  # Int[Array, " seq_len"],
):
    """
    Evaluate the model on a single pulse train.
    """
    labels_pred = model(pulse_train)
    return evaluate_labels(labels_pred, labels_true)


def evaluate_model_on_pulse_train_batch(
    model: Deinterleaver,
    pulse_train_batch,  # Float[Array, "batch_size seq_len feature_len"],
    labels_true_batch,  # Int[Array, "batch_size seq_len"],
):
    """
    Evaluate the model on a batch of pulse trains.
    Returns a dictionary of scores for each pulse train in the batch.
    """
    labels_pred_batch = model(pulse_train_batch)
    scores = []
    for labels_pred, labels_true in zip(
        labels_pred_batch, labels_true_batch, strict=False
    ):
        scores.append(evaluate_labels(labels_pred, labels_true))
    return {key: [score[key] for score in scores] for key in scores[0]}


def evaluate_model_on_dataset(model: Deinterleaver, dataloader: DataLoader):
    """
    Evaluate the model on a dataset.
    Returns the average scores for the dataset.
    """
    batch_scores = []
    for pulse_train_batch, labels_true_batch in tqdm(
        dataloader, desc="Evaluating model"
    ):
        batch_scores.append(
            evaluate_model_on_pulse_train_batch(
                model, pulse_train_batch, labels_true_batch
            )
        )
    all_scores = {
        key: [score for batch_score in batch_scores for score in batch_score[key]]
        for key in batch_scores[0]
    }
    return {key: sum(values) / len(values) for key, values in all_scores.items()}
