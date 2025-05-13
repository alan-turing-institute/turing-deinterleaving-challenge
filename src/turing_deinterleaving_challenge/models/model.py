from abc import ABC, abstractmethod


class Deinterleaver(ABC):
    """
    Abstract base class for deinterleaving models.
    This class defines the interface for deinterleaving models.
    The model takes as input a batch of pulse trainsand outputs
    a batch sequences of emitter IDs
    """

    @abstractmethod
    def __call__(
        self,
        data # Float[Array, "batch_size seq_len feature_len"],
    ): # -> Int[Array, "batch_size seq_len"]:
        pass
