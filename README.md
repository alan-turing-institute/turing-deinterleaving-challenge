# turing-deinterleaving-challenge

Welcome to the **turing-deinterleaving-challenge** repository! This project provides a comprehensive suite of utilities to support participants in the Alan Turing Institute's **Radar Pulse Train Deinterleaving Challenge**.

The challenge officially kicked off on **15th May 2025**.

The primary goal of this initiative is to foster collaboration within the radar deinterleaving community by addressing a common problem and **establishing a benchmark for comparing new methodologies.**

We encourage participation even if you join after the kick-off date. If you are taking part, it would be very appreciated if you **contact the organiser(s)** of the [Alan Turing Institute Machine Learning for Radio Frequency Interest Group](https://www.turing.ac.uk/research/interest-groups/machine-learning-radio-frequency-applications) - Dr Victoria Nockles <vnockles@turing.ac.uk>, for our records.

## Installation

### Recommended Environment Setup

Create a new virtual environment (e.g., using conda):

```bash
conda create -n deinterleaving_challenge python=3.11 pip
conda activate deinterleaving_challenge
```
*Note: The package requires Python >=3.11.*

### Install the Package

Then, install the `turing-deinterleaving-challenge` package using one of the following methods:

#### a. From PyPI (Recommended for most users)
 ```bash
python -m pip install turing-deinterleaving-challenge
```

#### b. From Source
```bash
git clone https://github.com/egunn-turing/turing-deinterleaving-challenge
cd turing-deinterleaving-challenge
python -m pip install .
```

#### c. In Development Mode
If you are contributing to the codebase or editing the Jupyter notebook `demo.ipynb`, install the package in development mode:

```bash
pip install -e ".[demo]"
```

Alternatively, to install the entire codebase in development mode:

```bash
pip install -e .
```

This allows you to make edits to the code without needing to reinstall the package. Note that any changes to imported classes/functions require you to **restart the Jupyter notebook kernel.**

## Usage

See the Jupyter notebook `demo.ipynb` for a detailed walkthrough of this codebase and the challenge data.

### Loading the Dataset

```python
from turing_deinterleaving_challenge import DeinterleavingChallengeDataset

# Load training data
train_dataset = DeinterleavingChallengeDataset(
    subset="train",      # OR val, test
    window_length=1000,  # Optional: process data in windows
    min_emitters=2,      # Optional: filter by minimum emitter count
    max_emitters=5       # Optional: filter by maximum emitter count
)

# Access data as a numpy array samples and labels for training, evaluation etc. 
data, labels = train_dataset[0]
```

### Visualise the Data

Use the `visualisation/visualisations.py` module to plot the Pulse Descriptor Word (PDW) data in a structured way.

### Directory Structure
```bash
└── src
    └── turing_deinterleaving_challenge
       ├── data
       │   ├── dataset.py
       │   ├── load.py
       │   └── structure.py
       ├── models
       │   ├── evaluate.py
       │   └── model.py
       └── visualisation
            └── visualisations.py
```

#### Important Class/Function Locations
* `data/dataset.py` contains the principal data class, `DeinterleavingChallengeDataset`.
* `data/load.py` defines a helper function `download_dataset` which downloads the challenge data from the Huggingface hub to a local directory, saving in the `.h5` format.
* `data/structure.py` defines the `PulseTrain` class, with various methods for saving and loading the data in scripts.

* `models/model.py` defines the Abstract Base Class that your model solution must wrap into.
* `models/evaluate.py` contains functions to evaluate your challenge model on the ground truth emitter labels. `evaluate_labels` in particular computes **V measure**, which is the principal evaluation metric of the challenge.

* `visualisation/visualisations.py` contains useful functions for plotting the PDW data in a structured way.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## License

Distributed under the terms of the [Apache license](LICENSE).

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/egunn-turing/turing-deinterleaving-challenge/workflows/CI/badge.svg
[actions-link]:             https://github.com/egunn-turing/turing-deinterleaving-challenge/actions
[pypi-link]:                https://pypi.org/project/turing-deinterleaving-challenge/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/turing-deinterleaving-challenge
[pypi-version]:             https://img.shields.io/pypi/v/turing-deinterleaving-challenge
<!-- prettier-ignore-end -->
