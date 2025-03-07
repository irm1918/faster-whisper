import logging
import os
import re

from typing import Optional

import huggingface_hub
import requests

from tqdm.auto import tqdm

_MODELS = (
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large-v1",
    "large-v2",
)


def get_assets_path():
    """
    Returns the path to the assets directory.
    
    Returns:
        str: The path to the assets directory.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def get_logger():
    """
    Returns the module logger.
    
    Returns:
        Logger: The logger of the module 'faster_whisper'.
    """
    return logging.getLogger("faster_whisper")


def download_model(
    size_or_id: str,
    output_dir: Optional[str] = None,
    local_files_only: bool = False,
    cache_dir: Optional[str] = None,
):
    """
    Downloads a CTranslate2 Whisper model from the Hugging Face Hub.
    
    The model is downloaded from https://huggingface.co/guillaumekln.
    
    Args:
        size_or_id (str): Size of the model to download (tiny, tiny.en, base, base.en, small, 
                          small.en, medium, medium.en, large-v1, or large-v2), or a 
                          CTranslate2-converted model ID from the Hugging Face Hub 
                          (e.g. guillaumekln/faster-whisper-large-v2).
        output_dir (str, optional): Directory where the model should be saved. If not set, the 
                                     model is saved in the cache directory.
        local_files_only (bool, optional): If True, avoid downloading the file and return the 
                                            path to the local cached file if it exists.
        cache_dir (str, optional): Path to the folder where cached files are stored.
    
    Returns:
        str: The path to the downloaded model.
    
    Raises:
        ValueError: If the model size is invalid.
    """
    if re.match(r".*/.*", size_or_id):
        repo_id = size_or_id
    else:
        if size_or_id not in _MODELS:
            raise ValueError(
                "Invalid model size '%s', expected one of: %s"
                % (size_or_id, ", ".join(_MODELS))
            )

        repo_id = "guillaumekln/faster-whisper-%s" % size_or_id

    allow_patterns = [
        "config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.*",
    ]

    kwargs = {
        "local_files_only": local_files_only,
        "allow_patterns": allow_patterns,
        "tqdm_class": disabled_tqdm,
    }

    if output_dir is not None:
        kwargs["local_dir"] = output_dir
        kwargs["local_dir_use_symlinks"] = False

    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir

    try:
        return huggingface_hub.snapshot_download(repo_id, **kwargs)
    except (
        huggingface_hub.utils.HfHubHTTPError,
        requests.exceptions.ConnectionError,
    ) as exception:
        logger = get_logger()
        logger.warning(
            "An error occured while synchronizing the model %s from the Hugging Face Hub:\n%s",
            repo_id,
            exception,
        )
        logger.warning(
            "Trying to load the model directly from the local cache, if it exists."
        )

        kwargs["local_files_only"] = True
        return huggingface_hub.snapshot_download(repo_id, **kwargs)


def format_timestamp(
    seconds: float,
    always_include_hours: bool = False,
    decimal_marker: str = ".",
) -> str:
    """
    Formats a timestamp in the format 'HH:MM:SS.sss'.
    
    Args:
        seconds (float): The timestamp in seconds.
        always_include_hours (bool, optional): If True, always include the hours in the output 
                                               even if it is zero. Defaults to False.
        decimal_marker (str, optional): The character to use as the decimal marker. 
                                        Defaults to '.'.
    
    Returns:
        str: The formatted timestamp.
    
    Raises:
        AssertionError: If the input timestamp is negative.
    """
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


class disabled_tqdm(tqdm):
    """
    A subclass of tqdm that disables the progress bar.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the disabled_tqdm class.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)
