import os
from dotenv import load_dotenv

from pathlib import Path
from huggingface_hub import snapshot_download

load_dotenv()
DATASET_ID = "egunn-turing/turing-deinterleaving-challenge"


def download_dataset(
    *, save_dir: Path | None = None, 
    subsets: str | list[str] | None = None, 
    hf_token: str | None = None
) -> None:
    """
    Download the dataset from Hugging Face Hub to a local directory.
    """
    if hf_token is None:
        print('Please ensure your .env file contains your HUGGING_FACE_TOKEN. Without may cause rate limiting issues.')
        hf_token = os.getenv("HUGGING_FACE_TOKEN")

    valid_subsets = ["train", "test", "validation"]
    if subsets is None:
        allow_patterns = ["*.h5"]
    else:
        if isinstance(subsets, str):
            subsets = [subsets]
        for subset in subsets:
            if subset not in valid_subsets:
                err = f"Invalid subset: {subset}. Valid subsets are: {valid_subsets}"
                raise ValueError(err)
        allow_patterns = list({f"{subset}/*.h5" for subset in subsets})

    return snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=save_dir,
        allow_patterns=allow_patterns,
        token=hf_token,
        max_workers=3,
    )
