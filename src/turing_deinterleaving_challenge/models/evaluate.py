from typing import Dict, List
import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    v_measure_score,
    matthews_corrcoef,
    f1_score,
    recall_score,
    precision_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing as mp

from .model import Deinterleaver
from .utils import fetch_async_result
from .. import PulseTrain

PENALTY_ALPHA = 1.


def cluster_wise_score(target: np.ndarray, cl: np.ndarray, score: str = "mcc") -> np.ndarray:
    score = score.lower()
    match score:
        case "mcc":
            score_fun = matthews_corrcoef
        case "recall":
            score_fun = recall_score
        case "precision":
            score_fun = precision_score
        case "f1":
            score_fun = f1_score
        case _:
            raise ValueError("Scoring function not recognized.")

    per_cluster_l = []
    for t in np.unique(target):
        score_l = []
        for c in np.unique(cl):
            c_mask = (c == cl).astype("int")
            score_l.append(score_fun(c_mask, (t == target).astype("int")))
        per_cluster_l.append(np.max(score_l))
    return np.min(per_cluster_l)



def evaluate_labels(
        labels_pred: np.ndarray,
        labels_true: np.ndarray,
        predict_ratio: float = 1.0,
) -> Dict[str, float]:
    homogeneity = homogeneity_score(labels_true, labels_pred)
    completeness = completeness_score(labels_true, labels_pred)
    v_measure = v_measure_score(labels_true, labels_pred)
    adjusted_rand_index = adjusted_rand_score(labels_true, labels_pred)
    adjusted_mutual_information = adjusted_mutual_info_score(labels_true, labels_pred)
    mcc = cluster_wise_score(labels_true, labels_pred, score="mcc")
    f1 = cluster_wise_score(labels_true, labels_pred, score="f1")
    penalty_score = predict_ratio**PENALTY_ALPHA

    return {
        "Homogeneity":penalty_score * homogeneity,
        "Completeness": penalty_score * completeness,
        "V-measure": penalty_score * v_measure,
        "Adjusted Rand Index": penalty_score * adjusted_rand_index,
        "Adjusted Mutual Information": penalty_score * adjusted_mutual_information,
        "MCC": penalty_score * mcc,
        "F1": penalty_score* f1,
        "discount": penalty_score
    }


def evaluate_model_on_pulse_train(
        model: Deinterleaver,
        pulse_train: np.ndarray,
        labels_true: np.ndarray,
):
    """
    Evaluate the model on a single pulse train.
    """
    labels_true = labels_true.flatten()
    labels_pred = model(pulse_train)
    ratio = 1.
    mask = np.ones(labels_pred.shape[0], dtype="bool")
    if model.default_label is not None:
        mask = labels_true != model.default_label
        ratio = mask.astype("float").sum() / float(mask.shape[0])
    return evaluate_labels(labels_pred[mask], labels_true[mask], predict_ratio=ratio)


def evaluate_model_on_pulse_train_batch(
    model: Deinterleaver,
    pulse_train_batch,  # Float[Array, "batch_size seq_len feature_len"],
    labels_true_batch,  # Int[Array, "batch_size seq_len"],
) -> Dict[str, List] | Dict:
    """
    Evaluate the model on a batch of pulse trains.
    Returns a dictionary of scores for each pulse train in the batch.
    """
    scores = []
    if len(pulse_train_batch.shape) == 2:
        n_pt, pt_dim = pulse_train_batch.shape
        pulse_train_batch = pulse_train_batch.reshape(1, n_pt, pt_dim)
        labels_true_batch = labels_true_batch.reshape(1, n_pt, 1)

    for pulse_train, labels in zip(pulse_train_batch, labels_true_batch, strict=False):
        scores.append(evaluate_model_on_pulse_train(model, pulse_train, labels))
    if len(scores) > 0:
        return {key: [score[key] for score in scores] for key in scores[0]}
    else:
        return {}


def evaluate_model_on_dataset(
        model: Deinterleaver,
        dataloader: DataLoader,
        n_jobs: int = 1,
        return_average: bool = False,
        max_eval: None | int = None
) -> Dict[str, float | List]:
    """
    Evaluate the model on a dataset. Returns the average scores for the dataset if the flag `return_average` is true.
    Otherwise, returns scores for all pulses in the dataset.

    Parameters
    ----------
    model: Deinterleaver
        Deinterleaver model. Must be child instance of Deinterleaver.
    dataloader: DataLoader
        Pulse train data loader
    return_average: bool
        If set, return the average over all pulse trains, otherwise return the entire list.
    """
    with mp.get_context("spawn").Pool(n_jobs) as pool:
        job_q = []
        for i_window, (pulse_train_batch, labels_true_batch) in tqdm(enumerate(dataloader), desc="Evaluating model"):
            if max_eval is not None and i_window >= max_eval:
                break
            if n_jobs < 2:
                job_q.append((
                    i_window,
                    evaluate_model_on_pulse_train_batch(model, pulse_train_batch, labels_true_batch)
                ))
            else:
                job_q.append((i_window, pool.apply_async(
                    evaluate_model_on_pulse_train_batch,
                    args=(model, pulse_train_batch, labels_true_batch)
                )))

        if n_jobs >= 2:
            pbar = tqdm(
                total=len(job_q),
                desc="Fetch results",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
            )
            job_q = fetch_async_result(job_q, process_bar=pbar)
        idc, batch_scores = zip(*sorted(job_q, key=lambda x: x[0]))

    if len(batch_scores) == 0:
        return {}

    all_scores = {
        key: [score for batch_score in batch_scores for score in batch_score[key]]
        for key in batch_scores[0]
    }
    if return_average:
        return {key: sum(values) / len(values) for key, values in all_scores.items()}
    else:
        return all_scores

