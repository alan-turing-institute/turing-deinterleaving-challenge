import os

from pathlib import Path
from huggingface_hub import snapshot_download

DATASET_ID = "alan-turing-institute/turing-synthetic-radar-dataset" 
HF_TOKEN_VAR_NAME = "HUGGING_FACE_TOKEN"
SUBSET_NAMES = ["train", "test", "validation"]
MODE_NAMES = ["stare", "scan"] 

def download_dataset(
    *, save_dir: Path | None = None, 
    subsets: str | list[str] | None = None, 
    modes: str | list[str] | None = None,
    hf_token: str | None = None,
    max_workers: int = 3,
    **kwargs
) -> None:
    """
    Download the dataset from Hugging Face Hub to a local directory.
    """
    if hf_token is None:
        from dotenv import load_dotenv
        load_dotenv()
        hf_token = os.getenv(HF_TOKEN_VAR_NAME)
    if hf_token is None:
        print(f'Please ensure your .env file contains your {HF_TOKEN_VAR_NAME}. Without may cause rate limiting issues.')

    if subsets is None:
        subsets = SUBSET_NAMES
    if isinstance(subsets, str):
        subsets = [subsets]
    for subset in subsets:
        if subset not in SUBSET_NAMES:
            err = f"Invalid subset: {subset}. Valid subsets are: {SUBSET_NAMES}"
            raise ValueError(err)

    if modes is None:
        modes = MODE_NAMES
    if isinstance(modes, str):
        modes = [modes]
    for mode in modes:
        if mode not in MODE_NAMES:
            err = f"Invalid mode: {mode}. Valid modes are: {MODE_NAMES}"
            raise ValueError(err)

    allow_patterns = list({
        f"{mode}/{subset}_{mode}/*.h5"
        for mode in modes
        for subset in subsets
    })

    return snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=save_dir,
        allow_patterns=allow_patterns,
        token=hf_token,
        max_workers=max_workers,
        **kwargs
    )