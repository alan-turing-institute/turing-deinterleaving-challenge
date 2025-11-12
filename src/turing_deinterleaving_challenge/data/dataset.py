from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from .load import download_dataset
from .structure import PulseTrain


class DeinterleavingChallengeDataset(Dataset):
    """
    A dataset class for the deinterleaving challenge.
    """

    def __init__(
        self,
        subset: str,
        window_length=None,
        local_path: Path | None = None,
        min_emitters: int | None = None,
        max_emitters: int | None = None,
        **kwargs
    ):
        # Validate and set basic parameters
        self._validate_subset(subset)
        self.subset = subset
        self.window_length = window_length
        self.min_emitters = min_emitters
        self.max_emitters = max_emitters

        # Set up dataset path
        self.local_path = local_path or download_dataset(subsets=subset, **kwargs)

        # Load and analyze data files
        self.data_files = self._get_sorted_data_files()
        # self.lengths stores pulse counts per file
        # self.file_total_emitters stores total unique emitters per file
        self.lengths, self.file_total_emitters = self._analyze_data_files()

        # Set dataset length and prepare windowing if needed
        # This method will now handle all sample validation and indexing.
        self._setup_dataset_length()

    def _validate_subset(self, subset):
        allowed_subsets = ["train", "test", "validation"]
        if subset not in allowed_subsets:
            err = f"Invalid subset: {subset}. Valid subsets are: {allowed_subsets}"
            raise ValueError(err)

    def _get_sorted_data_files(self):
        return sorted(
            (Path(self.local_path).glob(f"{self.subset}/*.h5")),
            key=lambda x: int(x.stem.split("_")[-1]),
        )

    def _analyze_data_files(self):
        lengths = []
        file_total_emitters = []  # Renamed from emitters
        for data_file in tqdm(self.data_files, desc="Analyzing files"):
            data = PulseTrain.load(data_file)
            lengths.append(data.data.shape[0])
            file_total_emitters.append(len(np.unique(data.labels)))
        return lengths, file_total_emitters

    def _is_valid_emitter_count(self, emitter_count: int) -> bool:
        """Checks if the given emitter count is within the configured min/max_emitters."""
        if self.min_emitters is not None and emitter_count < self.min_emitters:
            return False
        if self.max_emitters is not None and emitter_count > self.max_emitters:
            return False
        return True

    def _setup_dataset_length(self):
        if self.window_length is None:
            self.valid_file_indices = []  # Stores indices of files in self.data_files
            if self.data_files:  # Ensure there are files to process
                for i, _ in enumerate(self.data_files):
                    # A sample is the entire file. Check its total emitter count.
                    if self._is_valid_emitter_count(self.file_total_emitters[i]):
                        self.valid_file_indices.append(i)
            self.length = len(self.valid_file_indices)
            return

        # Validate window length (for windowed mode)
        if not isinstance(self.window_length, int) or self.window_length <= 0:
            raise ValueError(
                f"Invalid window length: {self.window_length}. Must be a positive integer."
            )

        if not self.lengths:  # No data files loaded or files have no pulses
            self.start_files_and_indices = []
            self.length = 0
            return  # Dataset is empty if no files or no pulses to window

        if self.window_length > max(self.lengths):
            raise ValueError(
                f"Invalid window length: {self.window_length}. Must be less than or equal to the maximum pulse train length ({max(self.lengths)})."
            )

        self.start_files_and_indices = []  # Stores (file_idx, window_start_pulse_idx)
        for file_idx, data_file_path in tqdm(
            enumerate(self.data_files),
            desc="Processing files for windows",
            total=len(self.data_files),
            unit="file",
        ):
            # Load the full PulseTrain object to access labels for window-based emitter counting
            pulse_train_full = PulseTrain.load(data_file_path)

            num_pulses_in_file = pulse_train_full.labels.shape[0]

            # Iterate over all possible full windows
            for window_num in range(num_pulses_in_file // self.window_length):
                start_pulse_idx = window_num * self.window_length
                end_pulse_idx = start_pulse_idx + self.window_length

                window_labels = pulse_train_full.labels[start_pulse_idx:end_pulse_idx]
                emitters_in_window = len(np.unique(window_labels))

                if self._is_valid_emitter_count(emitters_in_window):
                    self.start_files_and_indices.append((file_idx, start_pulse_idx))

        self.length = len(self.start_files_and_indices)

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return self.length

    def __getitem__(self, idx):
        """
        Returns the sample at the given index.
        """
        if idx < 0 or idx >= self.length:
            raise IndexError(
                f"Dataset index {idx} out of range for dataset of length {self.length}."
            )

        if self.window_length is None:
            # Non-windowed mode: idx refers to an index in self.valid_file_indices
            actual_file_idx = self.valid_file_indices[idx]
            data_file = self.data_files[actual_file_idx]
            pulse_train = PulseTrain.load(data_file)
            return pulse_train.data, pulse_train.labels  # Returns the .data attribute

        # Windowed mode: idx refers to an index in self.start_files_and_indices
        file_idx, start_pulse_idx = self.start_files_and_indices[idx]
        end_pulse_idx = start_pulse_idx + self.window_length

        data_file = self.data_files[file_idx]
        # PulseTrain.load_data_slicable returns an HDF5 dataset object for 'data'
        # which supports efficient slicing.
        data_h5_dataset = PulseTrain.load_data_slicable(data_file)

        return data_h5_dataset[start_pulse_idx:end_pulse_idx]
