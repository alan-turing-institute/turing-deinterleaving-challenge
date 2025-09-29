# Turing Deinterleaving Challenge

Welcome to the **turing-deinterleaving-challenge** repository! This project provides a comprehensive suite of utilities to support participants in the Alan Turing Institute's **Radar Pulse Train Deinterleaving Challenge**.

The challenge officially kicked off on **15th May 2025**.

The primary goal of this initiative is to foster collaboration within the radar deinterleaving community by addressing a common problem and **establishing a benchmark for comparing new methodologies.**

We encourage participation even if you join after the kick-off date. If you are taking part, it would be very appreciated if you **contact the organiser(s)** of the [Alan Turing Institute Machine Learning for Radio Frequency Interest Group](https://www.turing.ac.uk/research/interest-groups/machine-learning-radio-frequency-applications) - Dr Victoria Nockles <vnockles@turing.ac.uk>, for our records.

## Table of Contents

- [Technical Background](#technical-background)
  - [Radar Pulse Deinterleaving Problem](#radar-pulse-deinterleaving-problem)
  - [Pulse Descriptor Words (PDWs)](#pulse-descriptor-words-pdws)
  - [Deinterleaving Approaches](#deinterleaving-approaches)
  - [Challenge Context](#challenge-context)
- [Installation](#installation)
  - [Recommended Environment Setup](#recommended-environment-setup)
  - [Install the Package](#install-the-package)
- [Usage](#usage)
  - [Loading the Dataset](#loading-the-dataset)
  - [Visualise the Data](#visualise-the-data)
  - [Directory Structure](#directory-structure)
  - [Important Class/Function Locations](#important-classfunction-locations)
- [Contributing](#contributing)
- [License](#license)

## Technical Background

### Radar Pulse Deinterleaving Problem

Radar deinterleaving is a critical signal processing task in electronic warfare, surveillance, and radar signal intelligence applications. When multiple radar emitters operate simultaneously within the same electromagnetic environment, their transmitted pulses become interleaved in time, creating complex pulse trains that must be separated and attributed to their originating sources.

The radar pulse deinterleaving problem involves separating radar pulses from multiple unknown emitters present in a single recorded pulse train. This separation task is particularly challenging because:

- The number of active emitters is typically unknown a priori
- Pulse patterns may be irregular or adaptive
- Environmental factors introduce noise and measurement uncertainty
- Real-time processing constraints limit computational complexity


Let $X = \lbrace x_{1}, x_{2}, \dots,x_{n} \rbrace$ represent a pulse train containing n pulses from N unknown emitters. The deinterleaving task seeks to partition $X$ into N disjoint subsets:

$$X = \lbrace U_{1},\dots U_{N} \rbrace$$

where each subset $U_{i}$ contains all pulses originating from emitter i.

![PDW Schema](.assets/TA/Schema.png)

*Figure 1: Schematic illustration of the radar pulse deinterleaving problem, showing how interleaved emitters can be clustered by emitter with feature vector $x_{i}$.* 

### Pulse Descriptor Words (PDWs)

A Pulse Descriptor Word (PDW) is a multi-dimensional feature vector that characterizes the measurable parameters of a radar pulse. PDWs serve as the fundamental input for deinterleaving algorithms, providing quantitative descriptions of pulse characteristics.

The standard PDW parameters used in this challenge include:

#### Time of Arrival (ToA)

- **Definition**: Timestamp when the pulse leading edge is detected
- **Units**: Microseconds ($\mu s$)
- **Significance**: Enables temporal pattern analysis and pulse repetition interval (PRI) estimation

#### Centre Frequency (CF)

- **Definition**: Carrier frequency of the radar pulse
- **Units**: Megahertz (MHz) or Gigahertz (GHz)
- **Significance**: Primary discriminator for frequency-agile or fixed-frequency emitters

#### Pulse Width (PW)

- **Definition**: Duration of the pulse envelope
- **Units**: Microseconds ($\mu s$)
- **Significance**: Indicates radar type and operational mode

#### Angle of Arrival (AoA)

- **Definition**: Spatial direction from which the pulse arrives
- **Units**: Degrees (°) or radians
- **Significance**: Provides spatial discrimination between emitters

#### Amplitude/Power

- **Definition**: Peak or integrated power level of the received pulse
- **Units**: Decibels (dB) or linear scale
- **Significance**: Relates to emitter power and propagation distance

### Deinterleaving Approaches

#### Traditional Methods

- **Histogram-based**: Analyze statistical distributions of PDW parameters (PRI histograms, frequency clustering)
- **Sequence-based**: Exploit temporal ordering and pattern recognition (PRI sequence matching, Markov models)
- **Clustering**: Unsupervised learning approaches (K-means, DBSCAN, hierarchical clustering)

#### Modern Deep Learning Approaches

Recent advances leverage transformer architectures for deinterleaving using metric learning approaches:

- **Sequence-to-sequence models**: Process entire pulse trains simultaneously
- **Self-attention mechanisms**: Capture long-range dependencies between pulses
- **Triplet loss training**: Optimizes embedding similarity within emitters and dissimilarity between emitters
- **Synthetic data generation**: Creates controlled training scenarios with known ground truth

Performance is typically evaluated using clustering metrics such as **V-measure** (the primary evaluation metric for this challenge), Adjusted Mutual Information (AMI), and silhouette coefficients.

### Challenge Context

This challenge is inspired by recent research including "Radar Pulse Deinterleaving with Transformer Based Deep Metric Learning" (arXiv:2503.13476), which demonstrates transformer-based approaches achieving 0.882 adjusted mutual information score on synthetic radar pulse data using 5-dimensional PDWs.

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

### Important Class/Function Locations

- `data/dataset.py` contains the principal data class, `DeinterleavingChallengeDataset`.
- `data/load.py` defines a helper function `download_dataset` which downloads the challenge data from the Hugging Face hub to a local directory, saving in the `.h5` format.
- `data/structure.py` defines the `PulseTrain` class, with various methods for saving and loading the data in scripts.
- `models/model.py` defines the Abstract Base Class that your model solution must wrap into.
- `models/evaluate.py` contains functions to evaluate your challenge model on the ground truth emitter labels. `evaluate_labels` in particular computes **V measure**, which is the principal evaluation metric of the challenge.
- `visualisation/visualisations.py` contains useful functions for plotting the PDW data in a structured way.

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