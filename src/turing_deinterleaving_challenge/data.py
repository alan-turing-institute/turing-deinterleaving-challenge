from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Self, get_args, get_type_hints

import h5py
import numpy as np
from jaxtyping import Float

try:
    from pydantic import BaseModel

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = type("BaseModel", (), {})  # dummy class if pydantic not available


class PulseTrainType(Enum):
    """
    Enum class representing different types of pulse trains.

    Attributes:
        REAL (str): Represents a real pulse train.
        SYNTHETIC (str): Represents a synthetic pulse train.
        HYBRID (str): Represents a hybrid pulse train.
    """

    REAL = "real"
    SYNTHETIC = "synthetic"
    HYBRID = "hybrid"


class SliceablePulseTrainData:
    """
    A class to handle sliceable pulse train data from an HDF5 file.

    Attributes:
        load_path (Path): The path to the HDF5 file containing the data.

    Methods:
        __getitem__(sl: slice) -> tuple:
            Retrieves a slice of the data and labels from the HDF5 file.
            Args:
                sl (slice): The slice object specifying the range of data to retrieve.
            Returns:
                tuple: A tuple containing the sliced data and labels.
    """

    def __init__(self, load_path: Path):
        self.load_path = load_path

    def __getitem__(self, sl: slice):
        with h5py.File(self.load_path, "r") as f:
            data = f["data"][sl]
            labels = f["labels"][sl] if "labels" in f else None
        return data, labels


@dataclass
class PulseTrainMetadata:
    """
    A class to represent metadata for a pulse train.

    Attributes:
        feature_names (list[str]): A list of feature names.
        receiver (ReceiverParams): Parameters for the receiver.
        transmitters (list[TransmitterParams]): A list of parameters for the transmitters.
        type (PulseTrainType): The type of pulse train.
        description (str | None): A description of the pulse train. Defaults to None.
        collection_time_s (float | None): The collection time in seconds. Defaults to None.
        num_pulses (int | None): The number of pulses. Defaults to None.
        date_created (datetime | None): The date the metadata was created. Defaults to None.
    """

    feature_names: list[str]
    type: PulseTrainType
    receiver: Any = None  # Should be properly typed as ReceiverParams when available
    transmitters: Any = (
        None  # Should be properly typed as list[TransmitterParams] when available
    )
    description: str | None = None
    collection_time_s: float | None = None
    num_pulses: int | None = None
    date_created: datetime | None = None


@dataclass
class PulseTrainDataset:
    """
    PulseTrainDataset is a class that represents a dataset of pulse train data, including metadata, data, and optional labels.
    It provides methods to save and load the dataset to and from an HDF5 file.

    Attributes:
        metadata (PulseTrainMetadata): Metadata associated with the pulse train data.
        data (Float[np.ndarray, "seq_len num_features"]): The pulse train data.
        labels (Float[np.ndarray, "seq_len"] | None): Optional labels for the pulse train data.

    Methods:
        save(path: Path):
            Save the dataset to an HDF5 file at the specified path.

        _save_metadata(h5_group: h5py.Group, metadata: PulseTrainMetadata):
            Recursively save a dataclass or a list of dataclasses to an HDF5 group.

        load(cls, path: Path) -> Self:
            Load the dataset from an HDF5 file at the specified path.

        load_data_slicable(cls, path: Path) -> SliceablePulseTrainData:
            Load the dataset from an HDF5 file at the specified path and return a sliceable version of the data.

        _load_metadata(h5_group: h5py.Group, metadata_cls):
            Recursively load a dataclass or list of dataclasses from an HDF5 group.
    """

    metadata: PulseTrainMetadata
    data: Float[np.ndarray, "seq_len num_features"]
    labels: Float[np.ndarray, " seq_len"] | None

    def save(self, path: Path):
        """
        Save the dataset to an HDF5 file.

        Parameters:
        path (Path): The file path where the dataset will be saved.

        The method saves the following data to the HDF5 file:
        - Metadata: Stored in the "metadata" group.
        - Data: Stored in the "data" dataset with float32 data type and gzip compression.
        - Labels (optional): Stored in the "labels" dataset with int8 data type and gzip compression, if labels are present.
        """
        with h5py.File(path, "w") as f:
            f.create_group("metadata")
            PulseTrainDataset._save_metadata(f["metadata"], self.metadata)
            f.create_dataset(
                "data",
                data=self.data,
                dtype=np.float32,
                compression="gzip",
                compression_opts=9,
            )
            if self.labels is not None:
                f.create_dataset(
                    "labels",
                    data=self.labels,
                    dtype=np.int8,
                    compression="gzip",
                    compression_opts=9,
                )

    @staticmethod
    def _save_metadata(h5_group: h5py.Group, metadata: object):
        """
        Recursively save a dataclass, Pydantic BaseModel, or a list of either to an HDF5 group.

        This function handles various types of fields within the dataclass or model, including nested dataclasses/models,
        lists of dataclasses/models, primitive types (str, int, float, bool), lists, numpy arrays, Enums,
        and datetime objects. Unsupported types will raise a ValueError.

        Args:
            h5_group (h5py.Group): HDF5 group to save into.
            metadata (object): Dataclass, Pydantic model, or list of either to save.

        Raises:
            ValueError: If an unsupported type is encountered.
        """
        # Handle single dataclass instance
        if is_dataclass(metadata):
            for field in fields(metadata.__class__):  # type: ignore
                value = getattr(metadata, field.name)

                # Recursively handle nested dataclasses or lists of dataclasses
                if (
                    isinstance(value, list)
                    and value
                    and (
                        is_dataclass(value[0])
                        or (PYDANTIC_AVAILABLE and isinstance(value[0], BaseModel))
                    )
                ):
                    # Create a group for list of nested dataclasses
                    list_group = h5_group.create_group(field.name)

                    # Save each nested dataclass
                    for i, nested_instance in enumerate(value):
                        nested_subgroup = list_group.create_group(f"item_{i}")
                        PulseTrainDataset._save_metadata(
                            nested_subgroup, nested_instance
                        )

                # Handle other types
                elif is_dataclass(value) or (
                    PYDANTIC_AVAILABLE and isinstance(value, BaseModel)
                ):
                    # Recursively save nested dataclasses or models
                    nested_group = h5_group.create_group(field.name)
                    PulseTrainDataset._save_metadata(nested_group, value)
                elif isinstance(value, str | int | float | bool):
                    h5_group.attrs[field.name] = value
                elif isinstance(value, list):
                    # Convert simple lists to numpy arrays
                    h5_group.create_dataset(field.name, data=np.array(value, dtype="S"))
                elif isinstance(value, np.ndarray):
                    h5_group.create_dataset(field.name, data=value)
                elif isinstance(value, Enum):
                    h5_group.attrs[field.name] = value.value
                elif isinstance(value, datetime):
                    h5_group.attrs[field.name] = value.isoformat()
                else:
                    err = f"Unsupported type {type(value)} for field {field.name}"
                    raise ValueError(err)
        # Handle Pydantic model
        elif PYDANTIC_AVAILABLE and isinstance(metadata, BaseModel):
            # Get all fields in the model
            model_data = metadata.model_dump()

            # Store model class name for recreation during load
            h5_group.attrs["__model_class__"] = metadata.__class__.__name__

            for field_name, value in model_data.items():
                # Recursively handle nested models or lists of models
                if (
                    isinstance(value, list)
                    and value
                    and (
                        is_dataclass(value[0])
                        or (PYDANTIC_AVAILABLE and isinstance(value[0], BaseModel))
                    )
                ):
                    # Create a group for list of nested models
                    list_group = h5_group.create_group(field_name)

                    # Save each nested model
                    for i, nested_instance in enumerate(value):
                        nested_subgroup = list_group.create_group(f"item_{i}")
                        PulseTrainDataset._save_metadata(
                            nested_subgroup, nested_instance
                        )

                # Handle other types
                elif is_dataclass(value) or (
                    PYDANTIC_AVAILABLE and isinstance(value, BaseModel)
                ):
                    # Recursively save nested models
                    nested_group = h5_group.create_group(field_name)
                    PulseTrainDataset._save_metadata(nested_group, value)
                elif isinstance(value, str | int | float | bool):
                    h5_group.attrs[field_name] = value
                elif isinstance(value, list):
                    # Convert simple lists to numpy arrays
                    h5_group.create_dataset(field_name, data=np.array(value, dtype="S"))
                elif isinstance(value, np.ndarray):
                    h5_group.create_dataset(field_name, data=value)
                elif isinstance(value, Enum):
                    h5_group.attrs[field_name] = value.value
                elif isinstance(value, datetime):
                    h5_group.attrs[field_name] = value.isoformat()
                else:
                    err = f"Unsupported type {type(value)} for field {field_name}"
                    raise ValueError(err)

    @classmethod
    def load(cls, path: Path) -> Self:
        """
        Load a PulseTrainDataset from an HDF5 file.

        Args:
            path (Path): The path to the HDF5 file.

        Returns:
            PulseTrainDataset: An instance of PulseTrainDataset with loaded metadata, data, and labels (if available).

        Raises:
            AssertionError: If the loaded metadata is not an instance of PulseTrainMetadata.
        """
        with h5py.File(path, "r") as f:
            metadata = PulseTrainDataset._load_metadata(
                f["metadata"], PulseTrainMetadata
            )
            assert isinstance(metadata, PulseTrainMetadata)
            data = f["data"][:]
            if "labels" in f:
                labels = f["labels"][:]
        return cls(
            metadata=metadata,
            data=data,
            labels=labels,
        )

    @classmethod
    def load_data_slicable(cls, path: Path) -> SliceablePulseTrainData:
        """
        Load data from the specified path and return it as a SliceablePulseTrainData object.

        Args:
            path (Path): The path to the data file.

        Returns:
            SliceablePulseTrainData: An object containing the loaded data.
        """
        return SliceablePulseTrainData(path)

    @staticmethod
    def _load_metadata(h5_group: h5py.Group, metadata_cls):
        """
        Recursively load a dataclass, Pydantic model, or list of either from an HDF5 group

        Args:
            h5_group: HDF5 group to load from
            metadata_cls: Dataclass or Pydantic model type to reconstruct

        Returns:
            Reconstructed dataclass, Pydantic model, or list of either
        """
        # If the input is a group of nested dataclasses (list-like)
        if isinstance(h5_group, h5py.Group) and any(
            key.startswith("item_") for key in h5_group
        ):
            nested_list = []
            for key in sorted(h5_group.keys(), key=lambda x: int(x.split("_")[1])):
                # Ensure we pass the correct dataclass type
                nested_list.append(
                    PulseTrainDataset._load_metadata(h5_group[key], metadata_cls)
                )
            return nested_list

        # Check if this is a Pydantic model
        if PYDANTIC_AVAILABLE and issubclass(metadata_cls, BaseModel):
            kwargs = {}
            field_types = get_type_hints(metadata_cls)

            for field_name, field_type in field_types.items():
                if field_name in h5_group.attrs:
                    # Simple attributes
                    kwargs[field_name] = h5_group.attrs[field_name]
                elif field_name in h5_group:
                    # Datasets (like lists or nested models)
                    if isinstance(h5_group[field_name], h5py.Group):
                        # Determine the correct nested type
                        type_args = get_args(
                            field_type
                        )  # Extract generic args if available
                        nested_type = type_args[0] if type_args else field_type

                        # Handle nested models
                        if (
                            PYDANTIC_AVAILABLE
                            and isinstance(nested_type, type)
                            and issubclass(nested_type, BaseModel)
                        ) or is_dataclass(nested_type):
                            kwargs[field_name] = PulseTrainDataset._load_metadata(
                                h5_group[field_name], nested_type
                            )
                        else:
                            err = f"Unsupported nested type {nested_type} for field {field_name}"
                            raise ValueError(err)
                    else:
                        # Simple lists or arrays
                        kwargs[field_name] = h5_group[field_name][()]

            return metadata_cls(**kwargs)

        # Handle single dataclass
        if is_dataclass(metadata_cls):
            kwargs = {}
            for field in fields(metadata_cls):
                if field.name in h5_group.attrs:
                    # Simple attributes
                    kwargs[field.name] = h5_group.attrs[field.name]
                elif field.name in h5_group:
                    # Datasets (like lists or nested dataclasses)
                    if isinstance(h5_group[field.name], h5py.Group):
                        # Determine the correct nested type
                        type_args = get_args(
                            field.type
                        )  # Extract generic args if available
                        nested_type = type_args[0] if type_args else field.type

                        # Ensure the nested type is a dataclass before recursion
                        if is_dataclass(nested_type):
                            kwargs[field.name] = PulseTrainDataset._load_metadata(
                                h5_group[field.name], nested_type
                            )
                        # Handle fields of type Any or dict-like structures
                        elif nested_type == Any or str(nested_type).startswith(
                            "typing.Any"
                        ):
                            # For Any type, load as a dictionary
                            nested_dict = {}
                            nested_group = h5_group[field.name]

                            # Load attributes into the dictionary
                            for attr_name, attr_value in nested_group.attrs.items():
                                nested_dict[attr_name] = attr_value

                            # Load datasets into the dictionary
                            for dataset_name in nested_group.keys():
                                if isinstance(nested_group[dataset_name], h5py.Group):
                                    # Recursively load nested groups as dictionaries
                                    nested_dict[dataset_name] = (
                                        PulseTrainDataset._load_metadata(
                                            nested_group[dataset_name], dict
                                        )
                                    )
                                else:
                                    nested_dict[dataset_name] = nested_group[
                                        dataset_name
                                    ][()]

                            kwargs[field.name] = nested_dict
                        else:
                            err = f"Unsupported nested type {nested_type} for field {field.name}"
                            raise ValueError(err)
                    else:
                        # Simple lists or arrays
                        kwargs[field.name] = h5_group[field.name][()]

            return metadata_cls(**kwargs)

        # Special case for loading into a dictionary (when metadata_cls is dict)
        if metadata_cls is dict:
            result_dict = {}
            # Load attributes
            for attr_name, attr_value in h5_group.attrs.items():
                result_dict[attr_name] = attr_value
            # Load datasets
            for key in h5_group:
                if isinstance(h5_group[key], h5py.Group):
                    result_dict[key] = PulseTrainDataset._load_metadata(
                        h5_group[key], dict
                    )
                else:
                    result_dict[key] = h5_group[key][()]
            return result_dict

        err = f"Unsupported dataclass type {metadata_cls}"
        raise ValueError(err)
