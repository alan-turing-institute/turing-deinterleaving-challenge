# turing-deinterleaving-challenge

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

A set of utilities to support the Alan Turing Institute's  radar pulse train deinterleaving challenge, kicked off on the 15th May 2025. 

The purpose of this hackathon is to bring the radar deinterleaving community together to work on a common problem, **providing a benchmark by which new methods can be compared against.** As such, you are very welcome to join late, if you are taking part, it would be very appreciated if you **contact the organiser(s)** of the [Alan Turing Institute Machine Learning for Radio Frequency Interest Group](https://www.turing.ac.uk/research/interest-groups/machine-learning-radio-frequency-applications) - Dr Victoria Nockles <vnockles@turing.ac.uk>, for our records.

## Installation

### Recommended

Create a new virtual environment e.g., with conda

```bash
conda create -n deinterleaving_challenge python=3.11.12 pip
```

Then **install** the ```turing_deinterleaving_challenge``` package with one of the following:

### a. From the PyPI 
 ```bash
python -m pip install turing_deinterleaving_challenge
```

###  b. From source:
```bash
git clone https://github.com/egunn-turing/turing-deinterleaving-challenge
cd turing-deinterleaving-challenge
python -m pip install .
```

### c. In development mode (see below)
## Development

If you are just editing the jupyter notebook demo.ipynb,

 ```pip install -e ".[demo]"```

or to install the entire codebase in development mode.

```pip install -e .```

 This means that after edits to the code, you do not have to reinstall the package. Note that, regardless any edits you make to imported classes/functions you will have to **restart the Jupyter notebook kernel.**


## Usage

See the jupyter notebook ```demo.ipynb``` for a detailed walkthrough of this codebase & the challenge data.

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

# Access data samples and labels for training, evaluation etc.
data, labels = train_dataset[0]
```

### Visualise the data



### Directory structure:
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
#### Important class/function locations
* ```data/dataset.py``` contains the principal data class, ```DeinterleavingChallengeDataset```.
* ```data/load.py``` defines a helper function ```download_dataset``` which downloads the challenge data from the Huggingface hub to a local directory, saving in the .h5 format.
* ```data/structure.py``` defines the ```PulseTrain``` class, with various methods for saving & loading of the data in scripts.

* ```models/model.py``` defines the Abstract Base Class that your model solution must wrap into. 
* ```models/evaluate.py``` contains functions to evaluate your challenge model on the ground truth emitter labels. ```evaluate_labels``` in particular computes **V measure** which is the principal evaluation metric of the challenge.

* ```visualisation/visualisations.py``` contains useful functions for plotting the PDW data in a structured way.
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
